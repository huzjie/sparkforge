PY ?= python

all: compile test doctor

compile:
	$(PY) -m compileall -q sparkforge

test:
	$(PY) -m unittest discover -s tests -t .

doctor:
	$(PY) -m sparkforge.doctor

bench:
	$(PY) -m sparkforge.cli bench --config configs/default.yaml

code:
	$(PY) -m sparkforge.cli code --config configs/default.yaml --prompt "Fix the null-pointer in utils.py"

serve:
	$(PY) -m sparkforge.cli serve --config configs/default.yaml

.PHONY: all compile test doctor bench code serve
