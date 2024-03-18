CXX=g++
CXXFLAGS+=-std=c++17 -g -O0 -DDEBUG -Wno-unused-result -DMARRAY_ENABLE_ASSERTS -DDEBUG -Wno-unused-result 
CXXFLAGS+=-I../marray/marray 
CXXFLAGS+=-I../build/blis/include/firestorm -I../build/blis/include/haswell
CXXFLAGS+=-I../blis/frame/base
ARCH=$(shell uname -m)
ifeq ($(ARCH),arm64)
LDFLAGS=-g ../build/blis/lib/firestorm/libblis.a 
else
LDFLAGS=-g ../build/blis/lib/haswell/libblis.a
endif
#-fsanitize=address
#-ld_classic
#BLAS=-lblis
LIBS=-lm -ldl

exe=../bin/ltlt
testexe=../bin/test
objs=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))
objs:=$(filter-out ltlt.o test.o,$(objs))


.PHONY: all
all: $(exe) $(testexe)

-include $(deps)


.PHONY: clean
clean:
	rm -f $(objs) $(exe)

$(exe): $(objs) ltlt.o
	@mkdir -p $(dir $(exe))
	$(CXX) $(LDFLAGS) -o $(exe) $(objs) ltlt.o $(LIBS)

$(testexe): $(objs) test.o
	@mkdir -p $(dir $(testexe))
	$(CXX) $(LDFLAGS) -o $(testexe) $(objs) test.o $(LIBS)

%.o: %.cpp
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<
