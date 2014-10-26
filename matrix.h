#ifndef _MATRIX_H_

#define _MATRIX_H_

#include <stdio.h>

#include "synapse.h"
#include "layer.h"
#include "word_class.h"
#include "vocabulary.h"

class Layer;

class Matrix {
private:
	int _rows;
	int _columns;
	Synapse *_synapses;

public:
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
	void adjustRowWeights(int, real, const Layer &, const Layer &);
	void adjustColumnWeights(int, real, const Layer &, const Layer &);
	void adjustWeights(real, const Layer &, const Layer &);
	void adjustRowWeightsBeta2(int, real, real, const Layer &, const Layer &);
	void adjustColumnWeightsBeta2(int, real, real, const Layer &, const Layer &);
	void learnForWords(int, int, real, real, const Vocabulary &, const WordClass &, const Layer &, const Layer &);
	void learnForClasses(int, real, real, const Vocabulary &, const Layer &, const Layer &);
	int getRows() const { return _rows; };
	int getColumns() const { return _columns; };
	real getWeight(int which) const { return _synapses[which].weight; };
	void incrementWeight(int which, real amount) { _synapses[which].weight += amount; };
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