"""! @file swe_pinn/physics/common.py
@brief Shared dataclasses, network, nondimensional sampling, and physics wrappers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import tensorflow as tf

from swe_pinn.physics.quantum import QuantumLayer

DTYPE = tf.float64
ZERO = tf.constant(0.0, dtype=DTYPE)
ONE = tf.constant(1.0, dtype=DTYPE)

@dataclass(frozen=True)
class Domain1D:
    x_min: float
    x_max: float
    t_min: float
    t_max: float


def unit_interval(coord: tf.Tensor, lower: float, upper: float) -> tf.Tensor:
    """Map one coordinate from its physical interval onto [0, 1]."""
    lo = tf.cast(lower, DTYPE)
    span = tf.cast(upper - lower, DTYPE)
    return (coord - lo) / span

@dataclass
class Problem:
    name: str
    domain: Domain1D
    residual_fn: Callable
    output_fn: Callable
    exact_solution: Optional[Callable] = None
    to_dimensional: Optional[Callable] = None
    soft_constraint_loss_fn: Optional[Callable] = None
    n_outputs: int = 2


def _dense(width, name=None, activation=tf.nn.tanh):
    return tf.keras.layers.Dense(
        width,
        activation=activation,
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        bias_initializer="zeros",
        dtype=DTYPE,
        name=name,
    )


# Optuna may search a subset of these. full_quantum is the pure-quantum family:
# the field values are read out from the circuit, not from a classical trunk.
MODEL_FAMILIES = (
    "classical",
    "hybrid_quantum",
    "full_quantum",
)

# "once": one angle embedding, then input-independent layers.
# "reupload": the same features are embedded again at the start of every layer.
EMBEDDINGS = (
    "once",
    "reupload",
)


@tf.keras.utils.register_keras_serializable(package="Custom")
class PINN(tf.keras.Model):
    """Dual-tower PINN matching Desktop/2_dam_break (separate h and u paths).

    classical:
        each tower is a tanh MLP of ``depth`` layers of width ``width``.

    hybrid_quantum:
        the same MLP, plus the circuit correction on every forward pass
            tower = tower + decode(quantum(encode(tower)))
        The correction is not scaled by a trainable gate.

    full_quantum:
        pure quantum. There is no classical hidden trunk and no residual
        skip. Each tower is
            readout(quantum(encode(x, t)))
        where encode maps (x, t) onto ``n_qubits`` angles and readout is a
        linear map from Pauli-Z expectations to one scalar.
        ``embedding="once"`` uses a tanh encoder and one embedding.
        ``embedding="reupload"`` uses a linear encoder, and each layer
        embeds those angles again. ``width`` and ``depth`` are unused.
    """

    def __init__(
        self,
        width: int,
        depth: int,
        model_family: str = "classical",
        n_qubits: int = 0,
        q_layers: int = 0,
        embedding: str = "once",
        n_outputs: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = int(width)
        self.depth = int(depth)
        self.model_family = str(model_family)
        self.n_qubits = int(n_qubits)
        self.q_layers = int(q_layers)
        self.embedding = str(embedding)
        self.n_outputs = int(n_outputs)
        if self.n_outputs not in (2, 3):
            raise ValueError("n_outputs must be 2 or 3")

        if self.model_family == "hybrid_quantum" and self.n_qubits == 0:
            self.model_family = "classical"
            self.q_layers = 0

        if self.model_family == "classical":
            self.embedding = "once"
        elif self.embedding not in EMBEDDINGS:
            raise ValueError(
                f"Unknown embedding: {self.embedding}. Use one of {EMBEDDINGS}."
            )

        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(
                f"Unknown model_family: {self.model_family}. "
                f"Use one of {MODEL_FAMILIES}."
            )

        # When False, hybrid towers skip the QNode entirely (pure classical trunk).
        # Used for classical warmup so Optuna hybrid trials skip the circuit
        # for an initial fraction of epochs. full_quantum ignores this flag:
        # the circuit is the model.
        self.quantum_enabled = True

        # Back-compat aliases used by parameter counting / older code.
        self.hidden = None
        self.out = None
        self.pre_quantum = None
        self.quantum = None
        self.post_quantum_1 = None
        self.post_quantum_2 = None
        self.skip_proj = None
        self.alpha = None
        self.pre_quantum_h = None
        self.quantum_h = None
        self.post_quantum_h = None
        self.alpha_h = None
        self.pre_quantum_u = None
        self.quantum_u = None
        self.post_quantum_u = None
        self.alpha_u = None
        self.hidden_h = []
        self.hidden_u = []
        self.hidden_v = []
        self.out_h = None
        self.out_u = None
        self.out_v = None
        self.pre_quantum_v = None
        self.quantum_v = None
        self.post_quantum_v = None
        self.alpha_v = None

        if self.model_family == "full_quantum":
            if self.n_qubits < 1:
                raise ValueError("n_qubits must be >= 1 for full_quantum models")
            if self.q_layers < 1:
                raise ValueError("q_layers must be >= 1 for full_quantum models")
            self._build_full_quantum_towers()
        else:
            # Separate h and u towers (2_dam_break style).
            self.hidden_h = [_dense(self.width, name=f"h_hidden_{i}") for i in range(self.depth)]
            self.hidden_u = [_dense(self.width, name=f"u_hidden_{i}") for i in range(self.depth)]
            self.out_h = _dense(1, name="h_out", activation=None)
            self.out_u = _dense(1, name="u_out", activation=None)
            if self.n_outputs == 3:
                self.hidden_v = [_dense(self.width, name=f"v_hidden_{i}") for i in range(self.depth)]
                self.out_v = _dense(1, name="v_out", activation=None)
            if self.model_family == "hybrid_quantum":
                self._build_hybrid_quantum_filter()

        self.hidden = self.hidden_h
        self.out = self.out_h

    def _build_hybrid_quantum_filter(self):
        if self.n_qubits < 0:
            raise ValueError("n_qubits must be >= 0 for hybrid_quantum models")
        if self.q_layers < 0:
            raise ValueError("q_layers must be >= 0 for hybrid_quantum models")
        if self.n_qubits > 0 and self.q_layers < 1:
            raise ValueError("q_layers must be >= 1 for hybrid_quantum models")
        if self.n_qubits <= 0:
            return

        self.pre_quantum_h = self._angle_dense("pre_quantum_h")
        self.quantum_h = self._quantum_layer("quantum_h")
        self.post_quantum_h = _dense(self.width, name="post_quantum_h")

        self.pre_quantum_u = self._angle_dense("pre_quantum_u")
        self.quantum_u = self._quantum_layer("quantum_u")
        self.post_quantum_u = _dense(self.width, name="post_quantum_u")

        # Aliases so older quantum-param counters still see a quantum module.
        self.quantum = self.quantum_h
        self.pre_quantum = self.pre_quantum_h
        self.post_quantum_1 = self.post_quantum_h
        self.post_quantum_2 = self.post_quantum_h
        if self.n_outputs == 3:
            self._build_hybrid_quantum_tower("v")

    def _angle_dense(self, name: str):
        """Map the tower input onto one angle per qubit.

        Re-uploading uses a linear map so a coordinate weight can grow past
        one radian. A single embedding keeps the tanh map.
        """
        activation = None if self.embedding == "reupload" else tf.nn.tanh
        return _dense(self.n_qubits, name=name, activation=activation)

    def _quantum_layer(self, name: str):
        return QuantumLayer(
            n_qubits=self.n_qubits,
            q_layers=self.q_layers,
            embedding=self.embedding,
            name=name,
        )

    def _build_hybrid_quantum_tower(self, suffix: str):
        pre = self._angle_dense(f"pre_quantum_{suffix}")
        quantum = self._quantum_layer(f"quantum_{suffix}")
        post = _dense(self.width, name=f"post_quantum_{suffix}")
        setattr(self, f"pre_quantum_{suffix}", pre)
        setattr(self, f"quantum_{suffix}", quantum)
        setattr(self, f"post_quantum_{suffix}", post)

    def _build_full_quantum_towers(self):
        """Encode (x, t) into the circuit and read the field from its expectations."""
        self.pre_quantum_h = self._angle_dense("pre_quantum_h")
        self.quantum_h = self._quantum_layer("quantum_h")
        self.out_h = _dense(1, name="h_out", activation=None)

        self.pre_quantum_u = self._angle_dense("pre_quantum_u")
        self.quantum_u = self._quantum_layer("quantum_u")
        self.out_u = _dense(1, name="u_out", activation=None)
        if self.n_outputs == 3:
            self.pre_quantum_v = self._angle_dense("pre_quantum_v")
            self.quantum_v = self._quantum_layer("quantum_v")
            self.out_v = _dense(1, name="v_out", activation=None)

        self.quantum = self.quantum_h
        self.pre_quantum = self.pre_quantum_h

    def _apply_tower(self, inputs, hidden_layers, pre_q, quantum, post_q):
        z = inputs
        for layer in hidden_layers:
            z = layer(z)
        if quantum is not None and self.quantum_enabled:
            z = z + post_q(quantum(pre_q(z)))
        return z

    def _apply_full_quantum(self, inputs, pre_q, quantum, out):
        return out(quantum(pre_q(inputs)))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.cast(inputs, DTYPE)
        if self.model_family == "full_quantum":
            h = self._apply_full_quantum(x, self.pre_quantum_h, self.quantum_h, self.out_h)
            u = self._apply_full_quantum(x, self.pre_quantum_u, self.quantum_u, self.out_u)
            fields = [h, u]
            if self.n_outputs == 3:
                fields.append(
                    self._apply_full_quantum(x, self.pre_quantum_v, self.quantum_v, self.out_v)
                )
            return tf.concat(fields, axis=1)

        zh = self._apply_tower(
            x, self.hidden_h, self.pre_quantum_h, self.quantum_h, self.post_quantum_h
        )
        zu = self._apply_tower(
            x, self.hidden_u, self.pre_quantum_u, self.quantum_u, self.post_quantum_u
        )
        h = self.out_h(zh)
        u = self.out_u(zu)
        fields = [h, u]
        if self.n_outputs == 3:
            zv = self._apply_tower(
                x, self.hidden_v, self.pre_quantum_v, self.quantum_v, self.post_quantum_v
            )
            fields.append(self.out_v(zv))
        return tf.concat(fields, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "width": self.width,
                "depth": self.depth,
                "model_family": self.model_family,
                "n_qubits": self.n_qubits,
                "q_layers": self.q_layers,
                "embedding": self.embedding,
                "n_outputs": self.n_outputs,
            }
        )
        return cfg

def sample_interior(domain: Domain1D, n: tf.Tensor):
    x = tf.random.uniform((n, 1), domain.x_min, domain.x_max, dtype=DTYPE)
    t = tf.random.uniform((n, 1), domain.t_min, domain.t_max, dtype=DTYPE)
    return x, t


def sample_interior_grid(domain: Domain1D, nx: int, nt: int):
    """Fixed tensor-product collocation grid with ``nx`` by ``nt`` stations.

    Both axes include their endpoints. Flatten order is all times at the
    first ``x``, then all times at the next ``x``.
    """
    nx = int(nx)
    nt = int(nt)
    if nx < 2 or nt < 2:
        raise ValueError("nx and nt must both be >= 2")

    xs = np.linspace(domain.x_min, domain.x_max, nx, dtype=np.float64)
    ts = np.linspace(domain.t_min, domain.t_max, nt, dtype=np.float64)
    # Same flatten order as Generate_Data_scaled: for each x, all t.
    x_c = np.array([x for x in xs for t in ts], dtype=np.float64).reshape(-1, 1)
    t_c = np.array([t for x in xs for t in ts], dtype=np.float64).reshape(-1, 1)
    return tf.constant(x_c, dtype=DTYPE), tf.constant(t_c, dtype=DTYPE)

def outputs(model: PINN, problem: Problem, x: tf.Tensor, t: tf.Tensor, constraints: str):
    return problem.output_fn(model, x, t, constraints)

def residuals(model: PINN, problem: Problem, x: tf.Tensor, t: tf.Tensor, constraints: str):
    return problem.residual_fn(model, x, t, constraints)

def normalized_relative_weights(log_w_mass: float, log_w_mom: float):
    raw = np.exp(np.array([log_w_mass, log_w_mom], dtype=np.float64))
    norm = raw / raw.mean()
    return float(norm[0]), float(norm[1])
