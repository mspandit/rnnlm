#ifndef _LAYER_H_

#define _LAYER_H_

class Layer {
public:
	Neuron *_neurons;
	int _size;

	Layer() {
		_neurons = NULL;
		_size = 0;
	}
	~Layer() {
		if (_neurons != NULL) free(_neurons);
	}
	virtual void initialize(int);
	void copy(const Layer &);
	void clearActivation();
	void clearActivationRange(int, int);
	void clearError();
	void clear();
	void print(FILE *);
	void write(FILE *);
	void scan(FILE *);
	void read(FILE *);
	void setActivation(real);
	void receiveActivation(Layer &, int, Synapse []);
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
	void initialize(int);
	void backup(int);
	void restore(int);
};

#endif