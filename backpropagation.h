#ifndef _BACKPROPAGATION_H_

#define _BACKPROPAGATION_H_

#include "layer.h"
#include "synapse.h"

class Backpropagation {
private:
	static const int __history_buffer = 10;
    int *_word_history;
    int _time_steps; // amount of steps to propagate error back in time; default is 0 (equal to simple RNN)
    int _block; // # of time steps after which the error is backpropagated through time in block mode. Set to 1 to update at each time step.

public:
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
	void copy(const Layer &);
	int wordFromPast(int step) { return _word_history[step]; }
	void clearHistory();
	void setSteps(int steps) { _time_steps = steps; };
	int getSteps() const { return _time_steps; };
	void setBlock(int block) { _block = block; };
	int getBlock() const { return _block; };
};

#endif
