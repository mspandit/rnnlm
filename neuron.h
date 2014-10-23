#ifndef _NEURON_H_

#define _NEURON_H_

#include <stdio.h>

#include "types.h"

class Neuron {
public:
    real ac;		//actual value stored in neuron
    real er;		//error value in neuron, used by learning algorithm
	
	void copy(Neuron);
	void clear();
	void scanActivation(FILE *);
	void readActivation(FILE *);
	void applySigmoid();
};

#endif