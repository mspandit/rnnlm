#ifndef _MATRIX_H_

#define _MATRIX_H_

#include <stdio.h>

#include "synapse.h"
#include "neuron.h"
#include "vocabulary.h"

class Matrix {
public:
	Synapse *_synapses;
	int _rows;
	int _columns;

	Matrix() {
		_synapses = NULL;
	}

	~Matrix() {
		if (NULL != _synapses) free(_synapses);
	}
	void initialize(int rows, int columns);
	void copy(const Matrix &);
	void write(FILE *);
	void print(FILE *);
	void read(FILE *);
	void scan(FILE *);
	static real random(real, real);
	void randomize();
	void adjustRowWeights(int, real, Neuron [], Neuron []);
	void adjustColumnWeights(int, real, const Neuron [], const Neuron []);
	void adjustWeights(real, Neuron [], Neuron []);
	void adjustRowWeightsBeta2(int, real, real, Neuron [], Neuron []);
	void adjustColumnWeightsBeta2(int, real, real, const Neuron [], const Neuron []);
	void learnForWords(int, int, real, real, const Vocabulary &, const WordClass &, const Neuron [], const Neuron []);
	void learnForClasses(int, real, real, const Vocabulary &, const Neuron [], const Neuron []);
};

class MatrixBackup : public Matrix {
private:
	Matrix _backup;
	
public:
	void initialize(int, int);
	void backup();
	void restore();
};

#endif