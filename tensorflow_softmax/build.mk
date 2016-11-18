
# user friendly targets
softmax-all: softmax-lint softmax-test

softmax-test:

softmax-lint:
	pep8 --show-source --show-pep8 tensorflow_softmax/
