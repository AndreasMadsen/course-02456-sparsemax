
# -fPIC: means Position Independent Code
# -O2: level 2 optimizations
# -isystem: like -I for include but ignores warning from that source
OS=$(shell uname)
NVCC=nvcc
CXX=g++
CXXFLAGS=-std=c++11
CFLAGS=-fPIC -O2
NVCCFLAGS=-x cu -Xcompiler -fPIC
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


# square.so target
SQUARE_OBJS=kernel/custom_square.o
ifneq (, $(shell which nvcc))
SQUARE_OBJS+=kernel/custom_square.cu.o
endif

kernel/square.so: $(SQUARE_OBJS)
	$(CXX) $(LDFLAGS) -shared $^ -o $@

# generalized compile rule for c++ and CUDA
%.cu.o: %.cu.cc
	$(NVCC) $(CXXFLAGS) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f **/*.o
	rm -f **/*.so
	rm -rf kernel/__pycache__
