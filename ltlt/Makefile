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
           -I../build/blis/include/$(CONFIG) -I../blis/frame/base #-I../build/docopt.cpp/include/docopt/
LDFLAGS = -L../build/blis/lib/$(CONFIG) -Wl,-rpath,$(shell pwd)/../build/blis/lib/$(CONFIG) #-L../build/docopt.cpp/lib
LIBS = -lm -ldl -lblis #-ldocopt

ifeq ($(M3),true)
CXX = g++
CXXFLAGS += -O3 -march=native -fopenmp
else
CXX = clang++
CXXFLAGS += -g -O0 -DDEBUG -fsanitize=address -DMARRAY_ENABLE_ASSERTS
LDFLAGS += -g -fsanitize=address
endif

test_exe=../bin/ltlt_test
perf_exe=../bin/ltlt_perf
objs=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))
objs:=$(filter-out ltlt.o ltlt_test.o ltlt_perf.o,$(objs))

.PHONY: all
all: $(test_exe) $(perf_exe)

-include $(deps)

.PHONY: clean
clean:
	rm -f $(objs) $(exe) $(test_exe) $(perf_exe) ltlt.o ltlt_test.o ltlt_perf.o

$(test_exe): $(objs) ltlt_test.o
	@mkdir -p $(dir $(test_exe))
	$(CXX) $(LDFLAGS) -o $(test_exe) $(objs) ltlt_test.o $(LIBS)

$(perf_exe): $(objs) ltlt_perf.o
	@mkdir -p $(dir $(perf_exe))
	$(CXX) $(LDFLAGS) -o $(perf_exe) $(objs) ltlt_perf.o $(LIBS)

%.o: %.cpp
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<
