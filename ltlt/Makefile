ARCH = $(shell uname -m)
SYS = $(shell uname -s)

# determine the FLAGS for different arch we run
ifeq ($(ARCH),arm64)
CONFIG=firestorm
else ifeq ($(ARCH),x86_64)
ifeq ($(SYS),Darwin)
CONFIG=haswell
else
CONFIG=zen3
M3=true
endif
endif

CXXFLAGS = -std=c++17 -Wno-unused-result -Wno-unused-result -I../marray/marray \
           -I../build/blis/include/$(CONFIG) -I../blis/frame/base -I../build/docopt.cpp/include/docopt/ -DPROFILE
LDFLAGS = -L../build/blis/lib/$(CONFIG) -Wl,-rpath,$(shell pwd)/../build/blis/lib/$(CONFIG) -L../build/docopt.cpp/lib 
LIBS = -lm -ldl -lblis -ldocopt 
#/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0

ifeq ($(M3),true)
CXX = g++
CXXFLAGS += -O3 -march=native -fopenmp
LDFLAGS += -fopenmp
else
CXX = clang++
CXXFLAGS += -g -O0 -DDEBUG -Xclang -fopenmp -fsanitize=address -DMARRAY_ENABLE_ASSERTS -I/opt/homebrew/opt/libomp/include
LDFLAGS += -g -lomp -fsanitize=address -L/opt/homebrew/opt/libomp/lib
endif

debug_exe=../bin/ltlt_debug
test_exe=../bin/ltlt_test
perf_exe=../bin/ltlt_perf
benchmark_exe=../bin/ltlt_mbm
objs=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))
objs:=$(filter-out ltlt_debug.o ltlt_test.o ltlt_perf.o ltlt_mbm.o,$(objs))

.PHONY: all
all: $(test_exe) $(perf_exe) $(debug_exe) $(benchmark_exe)

-include $(deps)

.PHONY: clean
clean:
	rm -f $(objs) $(exe) $(test_exe) $(perf_exe) $(benchmark_exe) ltlt_test.o ltlt_perf.o ltlt_mbm.o ltlt_debug.o

$(benchmark_exe): $(objs) ltlt_mbm.o
	@mkdir -p $(dir $(benchmark_exe))
	$(CXX) -o $(benchmark_exe) $(objs) ltlt_mbm.o $(LDFLAGS) $(LIBS)

$(debug_exe): $(objs) ltlt_debug.o
	@mkdir -p $(dir $(debug_exe))
	$(CXX) -o $(debug_exe) $(objs) ltlt_debug.o $(LDFLAGS) $(LIBS)

$(test_exe): $(objs) ltlt_test.o
	@mkdir -p $(dir $(test_exe))
	$(CXX) -o $(test_exe) $(objs) ltlt_test.o $(LDFLAGS) $(LIBS)

$(perf_exe): $(objs) ltlt_perf.o
	@mkdir -p $(dir $(perf_exe))
	$(CXX) -o $(perf_exe) $(objs) ltlt_perf.o $(LDFLAGS) $(LIBS)

%.o: %.cpp
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<
