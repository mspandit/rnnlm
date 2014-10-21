#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "vocabulary.h"
#include "word_class.h"

void WordClass::initialize(const Vocabulary &vocab) {
	_words=(int **)calloc(_size, sizeof(int *));
	_word_count=(int *)calloc(_size, sizeof(int));
	_max_cn=(int *)calloc(_size, sizeof(int));
    
	for (int i = 0; i < _size; i++) {
		_word_count[i] = 0;
		_max_cn[i] = 10;
		_words[i] = (int *)calloc(_max_cn[i], sizeof(int));
	}
    
	for (int i = 0; i < vocab._size; i++) {
		int cl = vocab._words[i].class_index;
		_words[cl][_word_count[cl]++] = i;
		if (_word_count[cl] + 2 >= _max_cn[cl]) {
			_max_cn[cl] += 10;
			_words[cl] = (int *)realloc(_words[cl], _max_cn[cl] * sizeof(int));
		}
	}
}
