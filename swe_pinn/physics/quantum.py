"""! @file swe_pinn/physics/quantum.py
@brief Batched JAX PennyLane circuit for hybrid and pure-quantum PINNs.

The circuit runs the whole collocation batch in one compiled call. TensorFlow
still owns the trainable angles and the PDE tape. The bridge returns the
circuit vector-Jacobian product and a second custom gradient of that product,
so a residual built from spatial or time derivatives can update the angles.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

jax.config.update("jax_enable_x64", True)

DTYPE = tf.float64


def _require_pennylane():
    try:
        import pennylane as qml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for model_family='hybrid_quantum' or 'full_quantum'. "
            "Install it with: pip install pennylane"
        ) from exc
    return qml


def _compile_circuit(n_qubits: int, q_layers: int, embedding: str = "once"):
    """JIT a circuit vmapped over the batch. Weights are shared across samples.

    ``once`` embeds the features on the first gate, then each layer is an
    input-independent rotation and a CNOT ring. ``reupload`` applies that
    same feature vector again at the start of every layer, as a Y rotation,
    before the trainable rotation.
    """
    qml = _require_pennylane()
    if embedding not in ("once", "reupload"):
        raise ValueError(
            f"Unknown embedding {embedding!r}. Use 'once' or 'reupload'."
        )
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(features, weights):
        if embedding == "once":
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
        for layer in range(q_layers):
            if embedding == "reupload":
                qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            for wire in range(n_qubits):
                qml.Rot(
                    weights[layer, wire, 0],
                    weights[layer, wire, 1],
                    weights[layer, wire, 2],
                    wires=wire,
                )
            if n_qubits > 1:
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def batched(features, weights):
        expectations = jax.vmap(circuit, in_axes=(0, None))(features, weights)
        return jnp.stack(expectations, axis=1)

    batched = jax.jit(batched)

    def feature_vjp(features, weights, cotangent):
        _, pullback = jax.vjp(batched, features, weights)
        feature_grad, _ = pullback(cotangent)
        return feature_grad

    def weight_vjp(features, weights, cotangent):
        _, pullback = jax.vjp(batched, features, weights)
        _, weight_grad = pullback(cotangent)
        return weight_grad

    def feature_vjp_grad(features, weights, cotangent, upstream):
        """Derivative of the feature VJP, contracted with ``upstream``.

        This is the second derivative that a PDE residual needs. Holding the
        first Jacobian fixed would make every slope term drop out of the
        angle update.
        """
        _, pullback = jax.vjp(feature_vjp, features, weights, cotangent)
        return pullback(upstream)

    return (
        batched,
        jax.jit(feature_vjp),
        jax.jit(weight_vjp),
        jax.jit(feature_vjp_grad),
    )


def _as_jax(value):
    return jnp.asarray(np.asarray(value, dtype=np.float64))


def _as_tf(value):
    return tf.constant(np.asarray(value), dtype=DTYPE)


@tf.keras.utils.register_keras_serializable(package="Custom")
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits: int, q_layers: int, embedding: str = "once", **kwargs):
        super().__init__(dtype=DTYPE, **kwargs)
        self.n_qubits = int(n_qubits)
        self.q_layers = int(q_layers)
        self.embedding = str(embedding)
        if self.embedding not in ("once", "reupload"):
            raise ValueError(
                f"Unknown embedding {embedding!r}. Use 'once' or 'reupload'."
            )
        self._circuit = None
        self._feature_vjp = None
        self._weight_vjp = None
        self._feature_vjp_grad = None
        self.theta = None

    def build(self, input_shape):
        self.theta = self.add_weight(
            name="theta",
            # Three rotation angles per qubit per layer.
            # Quantum trainable parameters = 3 * q_layers * n_qubits.
            shape=(self.q_layers, self.n_qubits, 3),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            dtype=DTYPE,
            trainable=True,
        )
        (
            self._circuit,
            self._feature_vjp,
            self._weight_vjp,
            self._feature_vjp_grad,
        ) = _compile_circuit(self.n_qubits, self.q_layers, self.embedding)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, DTYPE)
        if inputs.shape.rank != 2:
            raise ValueError(
                f"QuantumLayer expects rank-2 inputs [batch, features]; got shape {inputs.shape}"
            )
        if inputs.shape[-1] != self.n_qubits:
            raise ValueError(
                f"QuantumLayer expected {self.n_qubits} features, got {inputs.shape[-1]}"
            )

        circuit = self._circuit
        feature_vjp = self._feature_vjp
        weight_vjp = self._weight_vjp
        feature_vjp_grad = self._feature_vjp_grad

        @tf.custom_gradient
        def feature_grad_op(features, weights, upstream):
            """Feature VJP as a TensorFlow op that can be differentiated again."""
            features_jax = _as_jax(features)
            weights_jax = _as_jax(weights)
            upstream_jax = _as_jax(upstream)
            value = _as_tf(feature_vjp(features_jax, weights_jax, upstream_jax))

            def grad(upstream_fg):
                feature_bar, weight_bar, upstream_bar = feature_vjp_grad(
                    features_jax,
                    weights_jax,
                    upstream_jax,
                    _as_jax(upstream_fg),
                )
                return _as_tf(feature_bar), _as_tf(weight_bar), _as_tf(upstream_bar)

            return value, grad

        @tf.custom_gradient
        def apply(features, weights):
            expectations = _as_tf(circuit(_as_jax(features), _as_jax(weights)))

            def grad(upstream):
                # The feature gradient stays on the tape. The outer PDE tape
                # differentiates it again, which is how a slope updates theta.
                feature_grad = feature_grad_op(features, weights, upstream)
                angle_grad = _as_tf(
                    weight_vjp(_as_jax(features), _as_jax(weights), _as_jax(upstream))
                )
                return feature_grad, angle_grad

            return expectations, grad

        # identity() is a Tensor, so the custom gradient does not capture the
        # Variable itself. The tape still routes this gradient back to theta.
        return apply(inputs, tf.identity(self.theta))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_qubits": self.n_qubits,
            "q_layers": self.q_layers,
            "embedding": self.embedding,
        })
        return cfg
