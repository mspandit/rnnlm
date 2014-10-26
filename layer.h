#ifndef _LAYER_H_

#define _LAYER_H_

#include "neuron.h"
#include "matrix.h"
#include "vocabulary.h"
#include "word_class.h"

class Backpropagation;
class Matrix;

class Layer {
private:
	Neuron *_neurons;
	int _size;

public:
	Layer() {
		_neurons = NULL;
		_size = 0;
	}
	~Layer() {
		if (_neurons != NULL) free(_neurons);
	}
	int getSize() const { return _size; }
	void setSize(int);
	virtual void initialize(int);
	void copy(const Layer &);
	void copyActivation(const Backpropagation &);
	void clear();
	void inputClear(int);
	void print(FILE *);
	void write(FILE *);
	void scan(FILE *);
	void read(FILE *);
	void setAllActivation(real);
	void setAllError(real);
	real getActivation(int neuron_index) const { return _neurons[neuron_index].ac; }
	void setActivation(int neuron_index, real activation) { _neurons[neuron_index].ac = activation; }
	void setActivationRange(int, int, real);
	real getError(int neuron_index) const { return _neurons[neuron_index].er; }
	void setError(int neuron_index, real error) { _neurons[neuron_index].er = error; }
	void setErrorRange(int, int, real);
	void receiveActivation(Layer &, int, const Matrix &);
	void incrementActivation(int, real);
	void incrementError(int, real);
	void applySigmoid();
	void deriveError();
	real maxActivation(const WordClass &, const Word &);
	double sumSigmoid(const WordClass &, const Word &, real);
	void setSigmoidActivation(const WordClass &, const Word &);
	void normalizeActivation(int);
};

class LayerBackup : public Layer {
	Layer _backups[2];

public:
	virtual void initialize(int);
	void backup(int);
	void restore(int);
};

class OutputLayer : public LayerBackup {
private:
	Vocabulary const *_vocab;
	WordClass const *_wordClass;

public:
	OutputLayer() { LayerBackup::LayerBackup(); _vocab = NULL; _wordClass = NULL; }
	using LayerBackup::initialize;
	virtual void initialize(const Vocabulary *, const WordClass *);
};

#endif