#include <stdlib.h>
#include <cfloat>

#include "fastexp.h"
#include "vocabulary.h"
#include "word_class.h"
#include "backpropagation.h"
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

void Layer::copyActivation(const Backpropagation &src) {
	for (int b = 0; b < _size; b++) 
		_neurons[b].ac = src.getActivation(b);		//restore hidden layer after bptt
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

void Layer::receiveActivation(Layer &src, int src_index, const Matrix &matrix) {
	for (int index = 0; index < _size; index++)
		_neurons[index].ac += src._neurons[src_index].ac * matrix.getWeight(src_index + index * src._size);
}

void Layer::applySigmoid()
{
	for (int layer_index = 0; layer_index < _size; layer_index++) 
		_neurons[layer_index].applySigmoid();
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

void Layer::setSigmoidActivation(const WordClass &wordClass, const Word &word) {
	real maxAc = maxActivation(wordClass, word);
	double sum = sumSigmoid(wordClass, word, maxAc);

	for (int c = 0; c < wordClass._word_count[word.class_index]; c++) {
		int a = wordClass._words[word.class_index][c];
		_neurons[a].ac=fasterexp(_neurons[a].ac-maxAc)/sum; //this prevents the need to check for overflow
	}
}

void Layer::normalizeActivation(int vocab_size) {
	double sum = 0.0;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here
	real max = -FLT_MAX;
	for (int layer2_index = vocab_size; layer2_index < _size; layer2_index++)
		if (_neurons[layer2_index].ac > max) max = _neurons[layer2_index].ac; //this prevents the need to check for overflow
	for (int layer2_index = vocab_size; layer2_index < _size; layer2_index++)
		sum += fasterexp(_neurons[layer2_index].ac - max);

	for (int layer2_index = vocab_size; layer2_index < _size; layer2_index++)
		_neurons[layer2_index].ac = fasterexp(_neurons[layer2_index].ac - max) / sum;
}

void Layer::clearActivationRange(int first_neuron, int num_neurons)
{
	for (int neuron_index = first_neuron; neuron_index < num_neurons; neuron_index++) 
		_neurons[neuron_index].ac = 0;
}

void Layer::clearErrorRange(int first_neuron, int num_neurons) {
	for (int a = first_neuron; a < num_neurons; a++)
		_neurons[a].er = 0;
}

void LayerBackup::initialize(int size) {
	Layer::initialize(size);
	
	for (int index = 0;  index < sizeof(_backups) / sizeof(Layer);  index++)
		_backups[index].initialize(size);
}

void LayerBackup::backup(int which_backup) {
	_backups[which_backup].copy(*this);
}

void LayerBackup::restore(int which_backup) {
	this->copy(_backups[which_backup]);
}