#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "neuron.h"
#include "synapse.h"
#include "matrix.h"

void Matrix::initialize(int rows, int columns) {
	_synapses = (Synapse *)calloc(rows * columns, sizeof(Synapse));
	_rows = rows;
	_columns = columns;
}

void Matrix::copy(const Matrix &src) {
	for (int b = 0; b < _rows; b++)
		for (int a = 0; a < _columns; a++) 
			_synapses[a + b * _columns].weight = src._synapses[a + b * _columns].weight;	
}

void Matrix::print(FILE *fo) {
	for (long long b = 0; b < _rows; b++) {
		for (int a = 0; a < _columns; a++) {
			_synapses[a + b * _columns].printWeight(fo);
		}
	}
}

void Matrix::write(FILE *fo) {
	for (long long b = 0; b < _rows; b++) {
		for (int a = 0; a < _columns; a++) {
			_synapses[a + b * _columns].writeWeight(fo);
		}
	}
}

void Matrix::scan(FILE *fi) {
	for (int b = 0; b < _rows; b++) {
		for (int a = 0; a < _columns; a++) {
			_synapses[a + b * _columns].scanWeight(fi);
		}
	}
}

void Matrix::read(FILE *fi) {
	for (int b = 0; b < _rows; b++) {
		for (int a = 0; a < _columns; a++) {
			_synapses[a + b * _columns].readWeight(fi);
		}
	}
}

/* static */
real Matrix::random(real min, real max)
{
	return rand()/(real)RAND_MAX*(max-min)+min;
}

void Matrix::randomize()
{
	for (int dest_index = 0; dest_index < _rows; dest_index++) 
		for (int src_index = 0; src_index < _columns; src_index++)
			_synapses[src_index + dest_index * _columns].weight = random(-0.1, 0.1) + random(-0.1, 0.1) + random(-0.1, 0.1);
}

void Matrix::adjustWeights(real alpha, Neuron row_neurons[], Neuron column_neurons[]) {
	for (int column = 0; column < _columns; column++) {
		for (int row = 0; row < _rows; row++) 
			_synapses[row + column * _rows].weight += alpha * column_neurons[column].er * row_neurons[row].ac;	//weight 1->c update
	}
}