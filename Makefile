SHELL := /bin/bash
VENV = ./venv

current_dir = $(shell pwd)

venv:
	python3.11 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip

requirements.txt: venv requirements.in
	rm -rf requirements.txt
	$(VENV)/bin/pip install pip-tools
	$(VENV)/bin/pip-compile || exit
	$(VENV)/bin/pip-sync || exit

clean:
	rm -f requirements.txt
	rm -f .coverage
	rm -f coverage.xml
	rm -rf htmlcov
	rm -rf venv
	find . -iname "*.pyc" -delete
