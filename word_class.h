#ifndef _WORD_CLASS_H_

#define _WORD_CLASS_H_

class WordClass {
public:
    int _size;
    int **_words;  // for words in each class, a list of their vocabulary indexes
    int *_word_count; // size of list in _words for each class
    int *_max_cn; // memory allocation in _words for each class
	
	~WordClass() {
	    for (int i = 0; i < _size; i++) free(_words[i]);
	    free(_words);
	    free(_word_count);
	    free(_max_cn);
	}
	void initialize(const Vocabulary &);
};

#endif