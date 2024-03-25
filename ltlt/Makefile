CXX=g++
CXXFLAGS+=-std=c++17 -O3 -Wno-unused-result -march=native
CXXFLAGS+=-I../marray/marray 
CXXFLAGS+=-I../blis/frame/base
LDFLAGS=

# add the bils.h with different machine
ARCH=$(shell uname -m)
ifeq ($(ARCH),arm64)
	CXXFLAGS+=-I../build/blis/include/firestorm
	LDFLAGS+=../build/blis/lib/firestorm/libblis.a
else ifeq ($(ARCH),x86_64)
	CXXFLAGS+=-I../build/blis/include/zen3
	LDFLAGS+=../build/blis/lib/zen3/libblis.a
else
	CXXFLAGS+=-I../build/blis/include/haswell
	LDFLAGS+=../build/blis/lib/haswell/libblis.a
endif


LIBS=-lm -ldl

exe=../bin/ltlt
test_exe=../bin/ltlt_test
objs=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))
objs:=$(filter-out ltlt.o ltlt_test.o,$(objs))

.PHONY: all
all: $(exe) $(test_exe)

-include $(deps)


.PHONY: clean
clean:
	rm -f $(objs) $(exe) $(test_exe) ltlt.o ltlt_test.o 

$(exe): $(objs) ltlt.o
	@mkdir -p $(dir $(exe))
	$(CXX) $(LDFLAGS) -o $(exe) $(objs) ltlt.o $(LIBS)

$(test_exe): $(objs) ltlt_test.o
	@mkdir -p $(dir $(test_exe))
	$(CXX) $(LDFLAGS) -o $(test_exe) $(objs) ltlt_test.o $(LIBS)

%.o: %.cpp
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<
