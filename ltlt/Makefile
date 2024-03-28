ARCH=$(shell uname -m)
# determine the FLAGS for different arch we run
ifeq ($(ARCH),arm64)
	CXX=clang++
	CXXFLAGS+=-std=c++17 -g -O0 -DDEBUG -fsanitize=address -Wno-unused-result -DMARRAY_ENABLE_ASSERTS -DDEBUG -Wno-unused-result 
	CXXFLAGS+=-I../marray/marray 
	CXXFLAGS+=-I../build/blis/include/firestorm
	CXXFLAGS+=-I../blis/frame/base
	CXXFLAGS+=-I../build/docopt.cpp/include/docopt/ 
	LDFLAGS=-g ../build/blis/lib/firestorm/libblis.a ../build/docopt.cpp/lib/libdocopt.a -lblas -fsanitize=address
	LIBS=-lm -ldl
else ifeq ($(ARCH),x86_64)
	CXX=g++
	CXXFLAGS+=-std=c++17 -O3 -Wno-unused-result -march=native
	CXXFLAGS+=-I../build/blis/include/zen3
	CXXFLAGS+=-I../marray/marray 
	CXXFLAGS+=-I../blis/frame/base
	LDFLAGS+=-L../build/blis/lib/zen3
	LDFLAGS=-Wl,-rpath,../build/blis/lib/zen3
else
	CXXFLAGS+=-I../build/blis/include/haswell
endif

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
