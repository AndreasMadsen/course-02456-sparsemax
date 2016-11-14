
# user friendly targets
sparsemax-python-all: sparsemax-python-test sparsemax-python-lint

sparsemax-python-test:
	nosetests --nologcapture -v -s tensorflow_python/test/test_*.py

sparsemax-python-lint:
	pep8 --show-source --show-pep8 ./tensorflow_python
