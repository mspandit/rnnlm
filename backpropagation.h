#ifndef _BACKPROPAGATION_H_

#define _BACKPROPAGATION_H_

class Backpropagation {
public:
    int _bptt;
    int _block;
    int *_history;
    Neuron *_neurons;
    Synapse *_synapses;
	int _rows;
	int _columns;
	
	Backpropagation() {
		_bptt = 0;
		_block = 10;
		_history = NULL;
		_neurons = NULL;
		_synapses = NULL;
	}
	
	~Backpropagation() {
		if (NULL != _history) free(_history);
		if (NULL != _neurons) free(_neurons);
		if (NULL != _synapses) free(_synapses);
	}
	void initialize(int, int);
	void reset();
	void shift(int, int);
};

#endif
