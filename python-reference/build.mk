
# user friendly targets
reference-all: test lint

reference-test:
	nosetests -v -s python-reference/test/test_*.py

reference-lint:
	pep8 --show-source --show-pep8 ./
