
# square.so target
tensorflow-square/kernel/custom_square.so: tensorflow-square/kernel/custom_square.o tensorflow-square/kernel/custom_square.cu.o

# user friendly targets
square-all: square-test square-lint

square-test: tensorflow-square/kernel/custom_square.so
	nosetests --nologcapture -v -s tensorflow-square/test/test_*.py

square-lint:
	pep8 --show-source --show-pep8 ./tensorflow-square
