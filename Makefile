.PHONY: test lint

#
# Compiler flags
#

# -fPIC: means Position Independent Code
# -O2: level 2 optimizations
# -isystem: like -I for include but ignores warning from that source
OS=$(shell uname)
NVCC=nvcc
CXX=g++
CXXFLAGS=-std=c++11
CFLAGS=-fPIC -O2 -Wall -Wextra
NVCCFLAGS=-x cu -Xcompiler -fPIC -Xcompiler -Wall -Xcompiler -Wextra
CPPFLAGS=-isystem $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS=

# OS specific rules
# -undefined dynamic_lookup: gives a linux like linking behaviour on MacOS
ifeq ($(OS), Darwin)
LDFLAGS+=-undefined dynamic_lookup
endif

# GPU on HPC
ifneq (, $(shell which nvcc))
LDFLAGS+=-L /appl/cuda/8.0/lib64 -L /appl/cudnn/v5.1-prod/lib64 -lcudart
CFLAGS+=-D GOOGLE_CUDA=1
NVCCFLAGS+=-D GOOGLE_CUDA=1
endif

#
# Build for .so, c++ and CUDA
#
%.so:
	$(CXX) $(LDFLAGS) -shared $^ -o $@

%.cu.o: %.cu.cc
ifneq (, $(shell which nvcc))
	$(NVCC) $(CXXFLAGS) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<
else
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<
endif

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

#
# User friendly rules
#
all: square-all sparsemax-all
test: square-test sparsemax-test
lint: square-lint sparsemax-lint

include tensorflow-sparsemax/build.mk
include tensorflow-square/build.mk

clean:
	rm -f **/**/*.o
	rm -f **/**/*.so
	rm -rf **/**/__pycache__
