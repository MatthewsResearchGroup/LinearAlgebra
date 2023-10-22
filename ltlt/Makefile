CXX=/opt/homebrew/Cellar/gcc/13.2.0/bin/g++-13
CXXFLAGS=-fopenmp -std=c++17 -g -O0 -DDEBUG -Wno-unused-result 
CXXFLAGS+=-I../marray/marray
LDFLAGS=-fopenmp
BLAS=-lblas
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
	$(CXX) $(LDFLAGS) -o $(exe) $(objs) $(BLAS) $(LIBS)

%.o: %.cxx
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<
