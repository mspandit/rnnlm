#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

#include "fastexp.h"
#include "types.h"
#include "neuron.h"
#include "synapse.h"
#include "vocabulary.h"
#include "word_class.h"
#include "layer.h"

void Layer::initialize(int size) {
	_size = size;
	_neurons = (Neuron *)calloc(_size, sizeof(Neuron));
}

void Layer::copy(const Layer &src) {
	for (int a = 0; a < _size; a++) {
		_neurons[a].copy(src._neurons[a]);
	}
}

void Layer::print(FILE *fo) {
	for (int a = 0; a < _size; a++) 
		fprintf(fo, "%.4f\n", _neurons[a].ac);
}

void Layer::write(FILE *fo) {
	for (int a = 0; a < _size; a++) {
		float fl = _neurons[a].ac;
		fwrite(&fl, sizeof(fl), 1, fo);
	}
}

void Layer::scan(FILE *fi) {
	for (int a = 0; a < _size; a++)
		_neurons[a].scanActivation(fi);
}

void Layer::read(FILE *fi) {
	for (int a = 0; a < _size; a++) {
		_neurons[a].readActivation(fi);
	}
}

void Layer::clear() {
	for (int a=0; a < _size; a++) {
		_neurons[a].clear();
	}
}

void Layer::setActivation(real activation) {
	for (int a = 0; a < _size; a++)
		_neurons[a].ac = 1.0;
}

void Layer::clearActivation() {
	for (int neuron_index = 0; neuron_index < _size; neuron_index++) 
		_neurons[neuron_index].ac = 0;	
}

void Layer::clearError()
{
	for (int neuron_index = 0; neuron_index < _size; neuron_index++) 
		_neurons[neuron_index].er = 0;
}

void Layer::receiveActivation(Layer &src, int src_index, Synapse matrix[]) {
	for (int index = 0; index < _size; index++)
		_neurons[index].ac += src._neurons[src_index].ac * matrix[src_index + index * src._size].weight;
}

void Layer::sigmoidActivation()
{
	for (int layer_index = 0; layer_index < _size; layer_index++) 
		_neurons[layer_index].sigmoidActivation();
}

void Layer::deriveError() {
	for (int a = 0; a < _size; a++) 
		_neurons[a].er = _neurons[a].er * _neurons[a].ac * (1 - _neurons[a].ac);
}

real Layer::maxActivation(const WordClass &wordClass, const Word &word) {
	real maxAc = -FLT_MAX;
	
	for (int c = 0; c < wordClass._word_count[word.class_index]; c++) {
		int a = wordClass._words[word.class_index][c];
		if (_neurons[a].ac > maxAc)
			maxAc = _neurons[a].ac;
	}
	return maxAc;
}

double Layer::sumSigmoid(const WordClass &wordClass, const Word &word, real maxAc) {
	double sum = 0.0;
	for (int c = 0; c < wordClass._word_count[word.class_index]; c++) {
		sum += fasterexp(_neurons[wordClass._words[word.class_index][c]].ac - maxAc);
	}
	return sum;
}

void Layer::setSigmoidActivation(const WordClass &wordClass, const Word &word, real maxAc, double sum) {
	for (int c = 0; c < wordClass._word_count[word.class_index]; c++) {
		int a = wordClass._words[word.class_index][c];
		_neurons[a].ac=fasterexp(_neurons[a].ac-maxAc)/sum; //this prevents the need to check for overflow
	}
}