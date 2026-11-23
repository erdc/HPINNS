MAIN = HPINNS_User_Manual

.PHONY: all clean

all: $(MAIN).pdf
	cp $(MAIN).pdf ../$(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex chapters/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
