CC = x86_64-linux-g++-4.6
WEIGHTTYPE = float
CFLAGS = -D WEIGHTTYPE=$(WEIGHTTYPE) -lm -O2 -Wall -funroll-loops -ffast-math
#CFLAGS = -lm -O2 -Wall

all: rnnlmlib.o options.o rnnlm

rnnlmlib.o : rnnlmlib.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c rnnlmlib.cpp

options.o : options.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c options.cpp

neuron.o : neuron.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c neuron.cpp

synapse.o : synapse.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c synapse.cpp

vocabulary.o : vocabulary.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c vocabulary.cpp

rnnlm.o : rnnlm.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c rnnlm.cpp

rnnlm : rnnlmlib.o options.o rnnlm.o neuron.o synapse.o vocabulary.o
	$(CC) $(CFLAGS) $(OPT_DEF) rnnlmlib.o options.o neuron.o synapse.o vocabulary.o rnnlm.o -o rnnlm

clean:
	rm -rf *.o rnnlm
