#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "neuron.h"
#include "synapse.h"
#include "backpropagation.h"

void Backpropagation::initialize(int rows, int columns) {
	_rows = rows;
	_columns = columns;
	if (_bptt > 0) {
		_history = (int *)calloc((_bptt + _block + 10), sizeof(int));
		_neurons = (Neuron *)calloc((_bptt + _block + 1) * columns, sizeof(Neuron));
		for (int a = 0; a < (_bptt + _block) * columns; a++) {
			_history[a] = -1;
			_neurons[a].clear();
		}
		_synapses = (Synapse *)calloc(_rows * _columns, sizeof(Synapse));
		if (_synapses == NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		}
	}
}

void Backpropagation::reset() {
	if (_bptt > 0) {
		for (int a = 1; a < _bptt + _block; a++) 
			_history[a] = 0;
		for (int a = _bptt + _block - 1; a > 1; a--) 
			for (int b = 0; b < _columns; b++) {
				_neurons[a * _columns + b].clear();
			}
	}
}

void Backpropagation::shift(int last_word, int layer_size) {
	if (_bptt > 0) {		//shift memory needed for bptt to next time step
		for (int a = _bptt + _block - 1; a > 0; a--)
			_history[a] = _history[a - 1];
		_history[0] = last_word;

		for (int a = _bptt + _block - 1; a > 0; a--) 
			for (int b = 0; b < layer_size; b++) {
				_neurons[a * layer_size + b].copy(_neurons[(a - 1) * layer_size + b]);
			}
	}
}