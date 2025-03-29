ARCH = $(shell uname -m)
SYS = $(shell uname -s)

# determine the FLAGS for different arch we run
ifeq ($(ARCH),arm64)
CONFIG=firestorm
else ifeq ($(ARCH),x86_64)
ifeq ($(SYS),Darwin)
CONFIG=haswell
else
CONFIG=x86_64
endif
endif

LIBBLIS = blis/lib/$(CONFIG)/libblis.a
LIBDOCOPT = docopt.cpp/libdocopt.a
CXXFLAGS = -std=c++20 -Wno-unused-result -Wno-unused-result -Imarray/marray \
           -Iblis/include/$(CONFIG) -Idocopt.cpp -DPROFILE
LIBS = -lm -ldl $(LIBBLIS) $(LIBDOCOPT)

CC = clang
CXX = clang++
CXXFLAGS += -O3 -march=native -fopenmp
LDFLAGS += -fopenmp

debug_exe=bin/ltlt_debug
perf_exe=bin/ltlt_perf
objs=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))
objs:=$(filter-out docopt.o ltlt_debug.o ltlt_test.o ltlt_perf.o ltlt_opt.o ltlt_mbm.o,$(objs))

.PHONY: all
all: $(perf_exe) $(debug_exe)

-include $(deps)

.PHONY: clean
clean:
	rm -f $(objs) $(debug_exe) $(perf_exe) ltlt_test.o ltlt_perf.o  ltlt_debug.o

.PHONY: distclean
distclean: clean
	$(MAKE) -C blis distclean
	$(MAKE) -C docopt.cpp distclean

blis/config.mk:
	@cd blis && ./configure CC=$(CC) CXX=$(CXX) -tomp --disable-shared $(CONFIG)

docopt.cpp/Makefile:
	@cd docopt.cpp && CC=$(CC) CXX=$(CXX) cmake .

$(LIBBLIS): blis/config.mk
	$(MAKE) -C blis

$(LIBDOCOPT): docopt.cpp/Makefile
	$(MAKE) -C docopt.cpp

$(debug_exe): $(objs) ltlt_debug.o
	@mkdir -p $(dir $(debug_exe))
	$(CXX) -o $(debug_exe) $(objs) ltlt_debug.o $(LDFLAGS) $(LIBS)

$(perf_exe): $(objs) ltlt_perf.o
	@mkdir -p $(dir $(perf_exe))
	$(CXX) -o $(perf_exe) $(objs) ltlt_perf.o $(LDFLAGS) $(LIBS)

%.o: %.cpp $(LIBBLIS) $(LIBDOCOPT)
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<

