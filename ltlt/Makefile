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

CXXFLAGS = -std=c++20 -Wno-unused-result -Wno-unused-result -I../marray/marray \
           -I../build/blis/include/$(CONFIG) -I../blis/frame/base -I../build/docopt.cpp/include/docopt/ -DPROFILE
LDFLAGS = -L../build/blis/lib/$(CONFIG) -Wl,-rpath,$(shell pwd)/../build/blis/lib/$(CONFIG) -L../build/docopt.cpp/lib
LIBS = -lm -ldl -lblis -ldocopt
#/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
#-lflame

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
perf_exe=../bin/ltlt_perf
#benchmark_exe=../bin/ltlt_mbm
objs=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))
objs:=$(filter-out ltlt_debug.o ltlt_test.o ltlt_perf.o ltlt_opt.o ltlt_mbm.o,$(objs))

.PHONY: all
#all: $(test_exe) $(perf_exe) $(debug_exe) $(benchmark_exe)
all: $(perf_exe) $(debug_exe)

-include $(deps)

.PHONY: clean
clean:
	rm -f $(objs) $(debug_exe) $(perf_exe) ltlt_test.o ltlt_perf.o  ltlt_debug.o

$(debug_exe): $(objs) ltlt_debug.o
	@mkdir -p $(dir $(debug_exe))
	$(CXX) -o $(debug_exe) $(objs) ltlt_debug.o $(LDFLAGS) $(LIBS)

$(perf_exe): $(objs) ltlt_perf.o
	@mkdir -p $(dir $(perf_exe))
	$(CXX) -o $(perf_exe) $(objs) ltlt_perf.o $(LDFLAGS) $(LIBS)

%.o: %.cpp
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<
