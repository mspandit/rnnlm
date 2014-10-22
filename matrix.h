#ifndef _MATRIX_H_

#define _MATRIX_H_

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
	void adjustWeights(real, Neuron [], Neuron []);
	void adjustWeightsBeta2(int, real, real, Neuron [], Neuron []);
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