
# sparsemax targets
tensorflow-sparsemax/kernel/sparsemax.so: tensorflow-sparsemax/kernel/sparsemax.o tensorflow-sparsemax/kernel/sparsemax_functor.o tensorflow-sparsemax/kernel/sparsemax_functor.cu.o

tensorflow-sparsemax/kernel/sparsemax_functor.o: tensorflow-sparsemax/kernel/sparsemax_functor.cc tensorflow-sparsemax/kernel/sparsemax_functor.h
tensorflow-sparsemax/kernel/sparsemax_functor.cu.o: tensorflow-sparsemax/kernel/sparsemax_functor.cu.cc tensorflow-sparsemax/kernel/sparsemax_functor.h

tensorflow-sparsemax/kernel/sparsemax_loss.so: tensorflow-sparsemax/kernel/sparsemax_loss.o tensorflow-sparsemax/kernel/sparsemax_loss.cu.o

tensorflow-sparsemax/kernel/sparsemax_loss.o: tensorflow-sparsemax/kernel/sparsemax_loss.cc tensorflow-sparsemax/kernel/sparsemax_loss.h
tensorflow-sparsemax/kernel/sparsemax_loss.cu.o: tensorflow-sparsemax/kernel/sparsemax_loss.cu.cc tensorflow-sparsemax/kernel/sparsemax_loss.h

# user friendly targets
sparsemax-all: sparsemax-test sparsemax-lint

sparsemax-test: tensorflow-sparsemax/kernel/sparsemax.so tensorflow-sparsemax/kernel/sparsemax_loss.so
	nosetests --nologcapture -v -s tensorflow-sparsemax/test/test_*.py

sparsemax-lint:
	pep8 --show-source --show-pep8 ./tensorflow-sparsemax
