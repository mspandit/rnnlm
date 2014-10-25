#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "backpropagation.h"

Backpropagation::Backpropagation() {
	_time_steps = 0;
	_block = 10;
	_word_history = NULL;
	_neurons = NULL;
	_synapses = NULL;
}

Backpropagation::~Backpropagation()  {
	if (NULL != _word_history) free(_word_history);
	if (NULL != _neurons) free(_neurons);
	if (NULL != _synapses) free(_synapses);
}

void Backpropagation::initialize(int rows, int columns) {
	_rows = rows;
	_columns = columns;
	if (_time_steps > 0) {
		_word_history = (int *)calloc((_time_steps + _block + __history_buffer), sizeof(int));
		_neurons = (Neuron *)calloc((_time_steps + _block + 1) * columns, sizeof(Neuron));
		for (int a = 0; a < (_time_steps + _block) * columns; a++) {
			_word_history[a] = -1;
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
	if (_time_steps > 0) {
		for (int a = 1; a < _time_steps + _block; a++) 
			_word_history[a] = 0;
		for (int a = _time_steps + _block - 1; a > 1; a--) 
			for (int b = 0; b < _columns; b++) {
				_neurons[a * _columns + b].clear();
			}
	}
}

void Backpropagation::shift(int last_word) {
	if (_time_steps > 0) {		//shift memory needed for bptt to next time step
		for (int a = (_time_steps + _block - 1); a > 0; a--)
			_word_history[a] = _word_history[a - 1];
		_word_history[0] = last_word;

		for (int a = (_time_steps + _block - 1); a > 0; a--) 
			for (int b = 0; b < _columns; b++) {
				_neurons[a * _columns + b].copy(_neurons[(a - 1) * _columns + b]);
			}
	}
}

void Backpropagation::adjustRowWeights(int row, real alpha, real activation, Neuron neurons[]) {
	for (int column = 0; column < _columns; column++)
		_synapses[row + column * _rows].weight += alpha * neurons[column].er * activation;
}

void Backpropagation::clearHistory() {
	if (_time_steps > 0) 
		for (int a = 0; a < _time_steps + _block; a++) // ignores __history_buffer (???) 
			_word_history[a] = 0;
}

void Backpropagation::copy(const Layer &layer) {
	for (int b = 0; b < layer._size; b++) 
		_neurons[b].copy(layer._neurons[b]);
}