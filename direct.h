#ifndef _DIRECT_H_

#define _DIRECT_H_

#include "layer.h"

typedef WEIGHTTYPE direct_t;	// ME weights

class Direct {
private:
	static const int MAX_NGRAM_ORDER = 20;
	static const unsigned int PRIMES[];
	static const unsigned int PRIMES_SIZE;

    direct_t *_synapses;		//direct parameters between input and output layer (similar to Maximum Entropy model parameters)
    long long _size;

    int _order;
    int _history[MAX_NGRAM_ORDER];
	
public:
	Direct() {
		_synapses = NULL;
		_size = 0;
		_order = 0;
	}
	~Direct() {
	    if (_synapses != NULL) free(_synapses);
	}
	long long getSize() const { return _size; };
	void setSize(long long);
	int getOrder() const { return _order; };
	void setOrder(int neworder) { _order = neworder; };
	void initialize(long long);
	void applyToClasses(Layer &, const Vocabulary &, int);
	void applyToWords(Layer &, int, const WordClass &);
	void learnForClasses(int, real, real, const Vocabulary &, const Layer &);
	void learnForWords(int, real, real, const Vocabulary &, const WordClass &, const Layer &);
	void clearHistory();
	void push(int);
	void print(FILE *);
	void write(FILE *);
	void scan(FILE *);
	void read(FILE *);
};

class DirectBackup : public Direct {
private:
	direct_t *_backup;
	
public:
	DirectBackup() {
		Direct();
		_backup = NULL;
	}
	~DirectBackup() {
		if (_backup != NULL) free(_backup);
	}
};

#endif