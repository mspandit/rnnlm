#ifndef _VOCABULARY_H_

#define _VOCABULARY_H_

#include <stdio.h>

#include "types.h"

class Word {
public:
    int cn;
    char word[MAX_STRING];

    real prob;
    int class_index;
	
	void set(char *);
};

class Vocabulary {
    int *_hash;
    int _hash_size;
    int _max_size;
    Word *_words;
    int _size;

public:
	~Vocabulary();
	void initialize(int, int, int);

    void sort();
    int search(char *word);
    int getHash(char *word);
	const Word & getWord(int index) const { return _words[index]; };
	int getSize() const { return _size; };
	void setSize(int newsize) { _size = newsize; };
    int add(char *word);
	void grow();
	void clear();
    int learnFromTrainFile(char *, int);
    void readWord(char *word, FILE *fin);
	void setClassIndexOld(int);
	void setClassIndexNew(int);
	void print(FILE *);
	void scan(FILE *);
};

#endif