
# sparsemax targets
tensorflow_sparsemax/kernel/sparsemax.so: tensorflow_sparsemax/kernel/sparsemax.o tensorflow_sparsemax/kernel/sparsemax_functor.o tensorflow_sparsemax/kernel/sparsemax_functor.cu.o

tensorflow_sparsemax/kernel/sparsemax_functor.o: tensorflow_sparsemax/kernel/sparsemax_functor.cc tensorflow_sparsemax/kernel/sparsemax_functor.h
tensorflow_sparsemax/kernel/sparsemax_functor.cu.o: tensorflow_sparsemax/kernel/sparsemax_functor.cu.cc tensorflow_sparsemax/kernel/sparsemax_functor.h

tensorflow_sparsemax/kernel/sparsemax_loss.so: tensorflow_sparsemax/kernel/sparsemax_loss.o tensorflow_sparsemax/kernel/sparsemax_loss.cu.o

tensorflow_sparsemax/kernel/sparsemax_loss.o: tensorflow_sparsemax/kernel/sparsemax_loss.cc tensorflow_sparsemax/kernel/sparsemax_loss.h
tensorflow_sparsemax/kernel/sparsemax_loss.cu.o: tensorflow_sparsemax/kernel/sparsemax_loss.cu.cc tensorflow_sparsemax/kernel/sparsemax_loss.h

# user friendly targets
sparsemax-all: sparsemax-test sparsemax-lint

sparsemax-test: tensorflow_sparsemax/kernel/sparsemax.so tensorflow_sparsemax/kernel/sparsemax_loss.so
	nosetests --nologcapture -v -s tensorflow_sparsemax/test/test_*.py

sparsemax-lint:
	pep8 --show-source --show-pep8 ./tensorflow_sparsemax
