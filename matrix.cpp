#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "neuron.h"
#include "synapse.h"
#include "matrix.h"

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
