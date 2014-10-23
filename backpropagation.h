#ifndef _BACKPROPAGATION_H_

#define _BACKPROPAGATION_H_

#include "neuron.h"
#include "synapse.h"

class Backpropagation {
private:
	static const int __history_buffer = 10;
    int *_word_history;

public:
    int _bptt;
    int _block;
    Neuron *_neurons;
    Synapse *_synapses;
	int _rows;
	int _columns;
	
	Backpropagation();
	~Backpropagation();
	void initialize(int, int);
	void reset();
	void shift(int);
	void adjustRowWeights(int, real, real, Neuron []);
	int wordFromPast(int step) { return _word_history[step]; }
	void clearHistory();
};

#endif
