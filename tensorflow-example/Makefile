
# -fPIC: means Position Independent Code
# -O2: level 2 optimizations
# -isystem: like -I for include but ignores warning from that source
OS=$(shell uname)
CXX=g++
CXXFLAGS=-std=c++11
CFLAGS=-fPIC -O2
CPPFLAGS=-isystem $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
LDFLAGS=

# OS specific rules
# -undefined dynamic_lookup: gives a linux like linking behaviour on MacOS
ifeq ($(OS), Darwin)
LDFLAGS+=-undefined dynamic_lookup
endif

# just linking no compiling
kernel/square.so: kernel/custom_square.o
	$(CXX) -D_GLIBCXX_USE_CXX11_ABI=0 $(LDFLAGS) -shared $^ -o $@

# generalized compile rule for c++
%.o: %.cc
	$(CXX) -D_GLIBCXX_USE_CXX11_ABI=0 $(CXXFLAGS) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

clean:
	rm **/*.o
	rm **/*.so
