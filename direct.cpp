#include <stdlib.h>

#include "vocabulary.h"
#include "word_class.h"
#include "layer.h"
#include "direct.h"

const unsigned int Direct::PRIMES[] = {
	108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
	407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
	782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807
};
const unsigned int Direct::PRIMES_SIZE = sizeof(Direct::PRIMES) / sizeof(unsigned int);

void Direct::push(int last_word) {
	for (int a = MAX_NGRAM_ORDER - 1; a > 0; a--) 
		_history[a] = _history[a-1];
	_history[0] = last_word;
}

void Direct::clearHistory() {
	for (int a = 0; a < MAX_NGRAM_ORDER; a++)
		_history[a] = 0;
}

void Direct::applyToClasses(Layer &layer, const Vocabulary &vocab, int layer2_size) {
	// Apply direct connections from earlier classes to current classes (???)
	if (_size>0) {
		unsigned long long hash[MAX_NGRAM_ORDER];	//this will hold pointers to _synapses that contains hash parameters
		for (int a = 0; a < _order; a++) hash[a] = 0;
		for (int a = 0; a < _order; a++) {
			if ((a > 0) && (_history[a - 1] == -1)) 
				break;	//if OOV was in history, do not use this N-gram feature and higher orders
			hash[a] = PRIMES[0] * PRIMES[1];
	    	    
			for (int b = 1; b <= a; b++) 
				hash[a] += PRIMES[(a * PRIMES[b] + b) % PRIMES_SIZE] * (unsigned long long)(_history[b - 1] + 1);	//update hash value based on words from the history
			hash[a] = hash[a] % (_size / 2);		//make sure that starting hash index is in the first half of _synapses (second part is reserved for history->words features)
		}
	
		for (int a = vocab.getSize(); a < layer2_size; a++) {
			for (int b = 0; b < _order; b++) 
				if (hash[b]) {
					layer.incrementActivation(a, _synapses[hash[b]]); //apply current parameter and move to the next one
					hash[b]++;
				} else 
					break;
		}
	}
}

void Direct::applyToWords(Layer &layer, int class_index, const WordClass &wordClass) {
	//apply direct connections from earlier words to current word (???)
	if (_size > 0) {
		unsigned long long hash[MAX_NGRAM_ORDER];
		for (int a = 0; a < _order; a++) hash[a] = 0;
		for (int a = 0; a < _order; a++) {
			if ((a > 0) && (_history[a - 1] == -1)) 
				break;
			hash[a] = PRIMES[0] * PRIMES[1] * (unsigned long long)(class_index + 1);
			for (int b = 1; b <= a; b++) 
				hash[a] += PRIMES[(a * PRIMES[b] + b) % PRIMES_SIZE] * (unsigned long long)(_history[b - 1] + 1);
			hash[a] = (hash[a] % (_size / 2)) + (_size / 2);
		}

		for (int c = 0; c < wordClass.wordCount(class_index); c++) {
			int a = wordClass.getWord(class_index, c);
			for (int b = 0; b < _order; b++) 
				if (hash[b]) {
					layer.incrementActivation(a, _synapses[hash[b]]);
					hash[b]++;
					hash[b] = hash[b] % _size;
				} else 
					break;
		}
	}
}

void Direct::learnForWords(int word, real alpha, real beta3, const Vocabulary &vocab, const WordClass &wordClass, const Layer &layer2) {
	if (_size > 0) {	//learn direct connections between words
		if (word != -1) {
			unsigned long long hash[MAX_NGRAM_ORDER];
	    
			for (int a=0; a<_order; a++) hash[a]=0;
	
			for (int a=0; a<_order; a++) {
				if (a>0) if (_history[a-1]==-1) break;
				hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(vocab.getWord(word).class_index+1);
				
				for (int b = 1; b <= a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(_history[b-1]+1);
				hash[a]=(hash[a]%(_size/2))+(_size)/2;
			}
	
			for (int c = 0; c < wordClass.wordCount(vocab.getWord(word).class_index); c++) {
				int a = wordClass.getWord(vocab.getWord(word).class_index, c);
	    
				for (int b=0; b<_order; b++) if (hash[b]) {
					_synapses[hash[b]] += alpha * layer2.getError(a) - _synapses[hash[b]] * beta3;
					hash[b]++;
					hash[b]=hash[b]%_size;
				} else break;
			}
		}
	}
}

void Direct::learnForClasses(int word, real alpha, real beta3, const Vocabulary &vocab, const Layer &layer2) {
	if (_size > 0) {
		//learn direct connections to classes
		//learn direct connections between words and classes
		unsigned long long hash[MAX_NGRAM_ORDER];
	
		for (int a=0; a<_order; a++) hash[a]=0;
	
		for (int a = 0; a<_order; a++) {
			if (a>0) if (_history[a-1]==-1) break;
			hash[a]=PRIMES[0]*PRIMES[1];
	    	    
			for (int b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(_history[b-1]+1);
			hash[a]=hash[a]%(_size/2);
		}
	
		for (int a=vocab.getSize(); a<layer2.getSize(); a++) {
			for (int b=0; b<_order; b++) if (hash[b]) {
				_synapses[hash[b]] += alpha * layer2.getError(a) - _synapses[hash[b]] * beta3;
				hash[b]++;
			} else break;
		}
	}
}

void Direct::initialize(long long newsize) {
	 _size = newsize;
	_synapses = (direct_t *)calloc((long long)_size, sizeof(direct_t));
	if (_synapses==NULL) {
		printf("Memory allocation for direct connections failed (requested %lld bytes)\n", (long long)_size * (long long)sizeof(direct_t));
		exit(1);
	}
	for (long long aa = 0; aa < _size; aa++) _synapses[aa]=0;
}

void Direct::setSize(long long newsize) {
	if (_size != newsize)
		initialize(newsize);		
}

void Direct::print(FILE *fo) {
	for (long long aa=0; aa<_size; aa++) {
		fprintf(fo, "%.2f\n", _synapses[aa]);
	}
}

void Direct::write(FILE *fo) {
	for (long long aa=0; aa<_size; aa++) {
		float fl = _synapses[aa];
		fwrite(&fl, sizeof(fl), 1, fo);
	}
}

void Direct::scan(FILE *fi) {
	double d;
	for (long long aa=0; aa<_size; aa++) {
		fscanf(fi, "%lf", &d);
		_synapses[aa]=d;
	}
}

void Direct::read(FILE *fi) {
	float fl;
	for (long long aa=0; aa<_size; aa++) {
		fread(&fl, sizeof(fl), 1, fi);
		_synapses[aa]=fl;
	}
}