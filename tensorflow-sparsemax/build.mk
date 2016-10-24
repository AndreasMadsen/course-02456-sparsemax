
# sparsemax targets
tensorflow-sparsemax/kernel/sparsemax.so: tensorflow-sparsemax/kernel/sparsemax.o

# user friendly targets
sparsemax-all: sparsemax-test sparsemax-lint

sparsemax-test: tensorflow-sparsemax/kernel/sparsemax.so
	nosetests --nologcapture -v -s tensorflow-sparsemax/test/test_*.py

sparsemax-lint:
	pep8 --show-source --show-pep8 ./tensorflow-sparsemax
