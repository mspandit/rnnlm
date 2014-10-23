#ifndef _SYNAPSE_H_

#define _SYNAPSE_H_

#include <stdio.h>

#include "types.h"
                
class Synapse {
public:
    real weight;	//weight of synapse
	
	void printWeight(FILE *);
	void writeWeight(FILE *);
	void readWeight(FILE *);
	void scanWeight(FILE *);
};

#endif
