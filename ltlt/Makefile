CXX=g++
CXXFLAGS+=-std=c++17 -g -O0 -DDEBUG -Wno-unused-result 
CXXFLAGS+=-I../marray/marray 
CXXFLAGS+=-I../build/blis/include/firestorm
CXXFLAGS+=-I../blis/frame/base
LDFLAGS=-g ../build/blis/lib/firestorm/libblis.a  
#-fsanitize=address
#-ld_classic
#BLAS=-lblis
LIBS=-lm -ldl

exe=../bin/ltlt
objs=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))


.PHONY: all
all: $(exe)

-include $(deps)


.PHONY: clean
clean:
	rm -f $(objs) $(exe)

$(exe): $(objs)
	@mkdir -p $(dir $(exe))
	$(CXX) $(LDFLAGS) -o $(exe) $(objs) $(LIBS)

%.o: %.cpp
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<
