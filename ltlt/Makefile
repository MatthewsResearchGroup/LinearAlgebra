CXX=clang++
CXXFLAGS+=-std=c++17 -g -O0 -DDEBUG -fsanitize=address -Wno-unused-result -DMARRAY_ENABLE_ASSERTS -DDEBUG -Wno-unused-result 
CXXFLAGS+=-I../marray/marray 
CXXFLAGS+=-I../build/blis/include/firestorm
CXXFLAGS+=-I../blis/frame/base
LDFLAGS=-g ../build/blis/lib/firestorm/libblis.a  -lblas -fsanitize=address
#-fsanitize=address
#-ld_classic
#BLAS=-lblis
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
