#include <stdio.h>
#include "fastexp.h"
#include "types.h"
#include "neuron.h"

void Neuron::copy(Neuron other) {
	ac = other.ac;
	er = other.er;
}

void Neuron::clear()
{
	ac = 0;
	er = 0;
}

void Neuron::scanActivation(FILE *fi) {
	double d;
	fscanf(fi, "%lf", &d);
	ac = d;
}

void Neuron::readActivation(FILE *fi) {
	float fl;
	fread(&fl, sizeof(fl), 1, fi);
	ac = fl;
}

void Neuron::applySigmoid() {
	if (ac > 50) ac = 50;  //for numerical stability
	if (ac < -50) ac = -50;  //for numerical stability
	ac = 1 / (1 + fasterexp(-ac));
}			
