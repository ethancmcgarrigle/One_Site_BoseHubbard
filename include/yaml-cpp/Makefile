sources := binary.cpp emit.cpp emitterstate.cpp exp.cpp node.cpp null.cpp parser.cpp scanscalar.cpp simplekey.cpp tag.cpp \
           convert.cpp emitfromevents.cpp emitterutils.cpp memory.cpp node_data.cpp ostream_wrapper.cpp regex_yaml.cpp scantag.cpp singledocparser.cpp \
           directives.cpp emitter.cpp exceptions.cpp nodebuilder.cpp nodeevents.cpp parse.cpp scanner.cpp scantoken.cpp stream.cpp

srcdir  := src
objdir  := obj
objects := $(addprefix $(objdir)/,$(sources:.cpp=.o))

libyaml-cpp.a: $(objects)
	rm -f $@
	ar cr $@ $^
	ranlib $@

$(objdir)/%.o: $(srcdir)/%.cpp | $(objdir)
	$(CXX) -c -std=c++11 $(CFLAGS) $(INCLUDE) -I./include/ -o $@ $<

$(objdir):
	mkdir -p $@

