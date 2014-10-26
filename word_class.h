#ifndef _WORD_CLASS_H_

#define _WORD_CLASS_H_

#include "vocabulary.h"

class WordClass {
private:
    int _size;
    int **_words;  // for words in each class, a list of their vocabulary indexes
    int *_word_count; // size of list in _words for each class
    int *_max_cn; // memory allocation in _words for each class

public:
	~WordClass() {
	    for (int i = 0; i < _size; i++) free(_words[i]);
	    free(_words);
	    free(_word_count);
	    free(_max_cn);
	}
	void initialize(const Vocabulary &);
	int getSize() const { return _size; }
	void setSize(int newsize) { _size = newsize; } // TODO
	int * getWords(int class_index) { return _words[class_index]; }
	int getWord(int class_index, int word_index) const { return _words[class_index][word_index]; }
	int firstWord(int class_index) const { return _words[class_index][0]; }
	int lastWord(int class_index) const { return _words[class_index][0] + _word_count[class_index]; }
	int wordCount(int class_index) const { return _word_count[class_index]; }
};

#endif