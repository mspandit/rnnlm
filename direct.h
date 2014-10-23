#ifndef _DIRECT_H_

#define _DIRECT_H_

typedef WEIGHTTYPE direct_t;	// ME weights

class Direct {
public:
	static const int MAX_NGRAM_ORDER = 20;
	static const unsigned int PRIMES[];
	static const unsigned int PRIMES_SIZE;

    direct_t *_synapses;		//direct parameters between input and output layer (similar to Maximum Entropy model parameters)
    long long _size;
    int _order;
    int _history[MAX_NGRAM_ORDER];
	
	Direct() {
		_synapses = NULL;
		_size = 0;
		_order = 0;
	}
	~Direct() {
	    if (_synapses != NULL) free(_synapses);
	}
	void applyToClasses(Neuron [], const Vocabulary &, int);
	void applyToWords(Neuron [], int, const WordClass &);
	void learnForClasses(int, real, real, const Vocabulary &, const Layer &);
	void learnForWords(int, real, real, const Vocabulary &, const WordClass &, const Layer &);
	void clearHistory();
	void push(int);
};

class DirectBackup : public Direct {
public:
	direct_t *_backup;
	
	DirectBackup() {
		Direct();
		_backup = NULL;
	}
	~DirectBackup() {
		if (_backup != NULL) free(_backup);
	}
};

#endif