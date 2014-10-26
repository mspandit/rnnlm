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

void Layer::setSize(int size) {
	if (_size != size)
		initialize(size);
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

void Layer::inputClear(int num_state_neurons) {
	for (int a = 0; a < (_size - num_state_neurons); a++)
		_neurons[a].clear();
	for (int a = _size - num_state_neurons; a < _size; a++) {   //last hidden layer is initialized to vector of 0.1 values to prevent unstability
		_neurons[a].ac=0.1;
		_neurons[a].er=0;
	}
}

void Layer::setAllActivation(real activation) {
	for (int a = 0; a < _size; a++)
		_neurons[a].ac = activation;
}

void Layer::setAllError(real error)
{
	for (int neuron_index = 0; neuron_index < _size; neuron_index++) 
		_neurons[neuron_index].er = error;
}

void Layer::receiveActivation(Layer &src, int src_index, const Matrix &matrix) {
	for (int index = 0; index < _size; index++)
		_neurons[index].ac += src._neurons[src_index].ac * matrix.getWeight(src_index + index * src._size);
}

void Layer::incrementActivation(int neuron_index, real increment) {
	_neurons[neuron_index].ac += increment;
}

void Layer::incrementError(int neuron_index, real increment) {
	_neurons[neuron_index].er += increment;
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

void Layer::setActivationRange(int first_neuron, int num_neurons, real activation)
{
	for (int neuron_index = first_neuron; neuron_index < num_neurons; neuron_index++) 
		_neurons[neuron_index].ac = activation;
}

void Layer::setErrorRange(int first_neuron, int num_neurons, real error) {
	for (int a = first_neuron; a < num_neurons; a++)
		_neurons[a].er = error;
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

void OutputLayer::initialize(const Vocabulary *vocab, const WordClass *wordClass) {
	LayerBackup::initialize(vocab->getSize() + wordClass->getSize());
	
	_vocab = vocab;
	_wordClass = wordClass;
}

void OutputLayer::setErrorsWord(int word) {
	for (int c = 0; c < _wordClass->wordCount(_vocab->getWord(word).class_index); c++) {
		setError(
			_wordClass->getWord(_vocab->getWord(word).class_index, c),
			(0 - getActivation(_wordClass->getWord(_vocab->getWord(word).class_index, c)))
		);
	}
	setError(word, 1 - getActivation(word));	//word part

	for (int a = _vocab->getSize(); a < getSize(); a++) {
		setError(a, (0 - getActivation(a)));
	}
	setError(
		_vocab->getSize() + _vocab->getWord(word).class_index,
		(1 - getActivation(_vocab->getSize() + _vocab->getWord(word).class_index))
	);	//class part
}

// Set activation for all words in the specified class
void OutputLayer::setClassActivation(int word_class, real activation) {
	for (int c = 0; c < _wordClass->wordCount(word_class); c++)
		setActivation(
			_wordClass->getWord(word_class, c), 
			activation
		);
}

void OutputLayer::setAllClassActivation(real activation) {
	setActivationRange(_vocab->getSize(), _size, activation);
}

void OutputLayer::normalizeClassActivation() {
	double sum = 0.0;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here
	real max = -FLT_MAX;
	for (int layer2_index = _vocab->getSize(); layer2_index < _size; layer2_index++)
		if (_neurons[layer2_index].ac > max) max = _neurons[layer2_index].ac; //this prevents the need to check for overflow
	for (int layer2_index = _vocab->getSize(); layer2_index < _size; layer2_index++)
		sum += fasterexp(_neurons[layer2_index].ac - max);

	for (int layer2_index = _vocab->getSize(); layer2_index < _size; layer2_index++)
		_neurons[layer2_index].ac = fasterexp(_neurons[layer2_index].ac - max) / sum;
}

real OutputLayer::maxActivation(int class_index) {
	real maxAc = -FLT_MAX;
	
	for (int c = 0; c < _wordClass->wordCount(class_index); c++) {
		int a = _wordClass->getWord(class_index, c);
		if (_neurons[a].ac > maxAc)
			maxAc = _neurons[a].ac;
	}
	return maxAc;
}

double OutputLayer::sumSigmoid(int class_index, real maxAc) {
	double sum = 0.0;
	for (int c = 0; c < _wordClass->wordCount(class_index); c++) {
		sum += fasterexp(_neurons[_wordClass->getWord(class_index, c)].ac - maxAc);
	}
	return sum;
}

void OutputLayer::setClassSigmoidActivation(int class_index) {
	real maxAc = maxActivation(class_index);
	double sum = sumSigmoid(class_index, maxAc);

	for (int c = 0; c < _wordClass->wordCount(class_index); c++) {
		int a = _wordClass->getWord(class_index, c);
		_neurons[a].ac = fasterexp(_neurons[a].ac - maxAc)/sum; //this prevents the need to check for overflow
	}
}
