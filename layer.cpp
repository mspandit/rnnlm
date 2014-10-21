#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "neuron.h"
#include "synapse.h"
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
