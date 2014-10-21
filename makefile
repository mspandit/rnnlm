CC = x86_64-linux-g++-4.6
WEIGHTTYPE = float
CFLAGS = -D WEIGHTTYPE=$(WEIGHTTYPE) -lm -O2 -Wall -funroll-loops -ffast-math
#CFLAGS = -lm -O2 -Wall

all: rnnlm

%.o : %.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c $< -o $@

rnnlm : rnnlmlib.o options.o rnnlm.o neuron.o synapse.o vocabulary.o layer.o matrix.o
	$(CC) $(CFLAGS) $(OPT_DEF) rnnlmlib.o options.o neuron.o synapse.o vocabulary.o rnnlm.o layer.o matrix.o -o $@

clean:
	rm -rf *.o rnnlm
