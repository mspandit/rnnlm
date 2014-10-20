#include <stdio.h>

#include "types.h"
#include "synapse.h"


void Synapse::printWeight(FILE *fo) {
	fprintf(fo, "%.4f\n", weight);
}

void Synapse::writeWeight(FILE *fo) {
	float fl = weight;
	fwrite(&fl, sizeof(fl), 1, fo);
}

void Synapse::scanWeight(FILE *fi) {
	double d;
	fscanf(fi, "%lf", &d);
	weight=d;
}

void Synapse::readWeight(FILE *fi) {
	float fl;
	fread(&fl, sizeof(fl), 1, fi);
	weight=fl;
}
