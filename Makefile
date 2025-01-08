# export ENABLE_PJRT_COMPATIBILITY=1 # enable compatibility with jaxlib # https://developer.apple.com/metal/jax/ 

VENV := venv
TESTS_FILES := $(wildcard *_test.py)

all: venv

$(VENV)/bin/activate: requirements.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	./$(VENV)/bin/python3 -m pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt

venv: $(VENV)/bin/activate 

test: $(TESTS_FILES) venv
	test -d $(TESTS_FILES) || $(VENV)/bin/python3 -m unittest discover . -p='*_test.py'

run: venv
	./$(VENV)/bin/python3 main.py

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete

.PHONY: run clean all venv