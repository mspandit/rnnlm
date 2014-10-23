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

public:
    Word *_words;
    int _size;
	
	~Vocabulary();
	void initialize(int, int, int);

    void sort();
    int search(char *word);
    int getHash(char *word);
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