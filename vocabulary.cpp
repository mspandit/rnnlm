#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "vocabulary.h"

void Vocabulary::readWord(char *word, FILE *fin)
{
	int a=0, ch;

	while (!feof(fin)) {
		ch=fgetc(fin);

		if (ch==13) continue;

		if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
			if (a>0) {
				if (ch=='\n') ungetc(ch, fin);
				break;
			}

			if (ch=='\n') {
				strcpy(word, (char *)"</s>");
				return;
			}
			else continue;
		}

		word[a]=ch;
		a++;

		if (a>=MAX_STRING) {
			//printf("Too long word found!\n");   //truncate too long words
			a--;
		}
	}
	word[a]=0;
}

int Vocabulary::getHash(char *word)
{
	unsigned int hash, a;
    
	hash=0;
	for (a=0; a<strlen(word); a++) hash=hash*237+word[a];
	hash = hash % _hash_size;
    
	return hash;
}

int Vocabulary::search(char *word)
{
	int a;
	unsigned int hash;
    
	hash = getHash(word);
    
	if (_hash[hash]==-1) return -1;
	if (!strcmp(word, _words[_hash[hash]].word)) return _hash[hash];
    
	for (a=0; a<_size; a++) {				//search in vocabulary
		if (!strcmp(word, _words[a].word)) {
			_hash[hash]=a;
			return a;
		}
	}

	return -1;							//return OOV if not found
}

void Word::set(char *the_word) {
	strcpy(word, the_word);
	cn = 0;
}

int Vocabulary::add(char *word)
{
	unsigned int hash;

	_words[_size++].set(word);

	if (_size+2>=_max_size) {        //reallocate memory if needed
		_max_size+=100;
		_words = (Word *)realloc(_words, _max_size * sizeof(Word));
	}
    
	hash=getHash(word);
	_hash[hash]=_size-1;

	return _size-1;
}

void Vocabulary::sort()
{
	int a, b, max;
	Word swap;
    
	for (a=1; a < _size; a++) {
		max=a;
		for (b=a+1; b< _size; b++) if (_words[max].cn<_words[b].cn) max=b;

		swap=_words[max];
		_words[max]=_words[a];
		_words[a]=swap;
	}
}

void Vocabulary::clear() {
	for (int a = 0; a < _hash_size; a++) _hash[a]=-1;
	_size=0;	
}

void Vocabulary::initialize(int max_size, int size, int hash_size) {
	_hash_size = hash_size;
	_hash=(int *)calloc(_hash_size, sizeof(int));
	_max_size = max_size;
	_size = size;
	_words=(Word *)calloc(_max_size, sizeof(Word));
}

Vocabulary::~Vocabulary() {
	free(_hash);
	free(_words);
}

int Vocabulary::learnFromTrainFile(char *train_file, int debug_mode)    //assumes that vocabulary is empty
{
	char word[MAX_STRING];
	FILE *fin;
	int a, i, train_wcn;

	fin=fopen(train_file, "rb");
	
	add((char *)"</s>");

	train_wcn=0;
	while (1) {
		readWord(word, fin);
		if (feof(fin)) break;
        
		train_wcn++;

		i=search(word);
		if (i==-1) {
			a=add(word);
			_words[a].cn=1;
		} else _words[i].cn++;
	}

	sort();

	if (debug_mode>0) {
		printf("Vocab size: %d\n", _size);
		printf("Words in train file: %d\n", train_wcn);
	}
    
	fclose(fin);
	return train_wcn;
}

void Vocabulary::setClassIndexOld(int class_size) {
	int b = 0;
	int a = 0;
	double df = 0;
	
	for (int i = 0; i < _size; i++)
		b += _words[i].cn;

	for (int i = 0; i < _size; i++) {
		df += _words[i].cn / (double)b;
		if (df > 1) df = 1;
		if (df > (a + 1) / (double)class_size) {
			_words[i].class_index = a;
			if (a < class_size - 1)
				a++;
		} else {
			_words[i].class_index = a;
		}
	}
}

void Vocabulary::setClassIndexNew(int class_size) {
	int b = 0;
	int a = 0;
	double dd = 0;
	double df = 0;

	for (int i = 0; i < _size; i++)
		b += _words[i].cn;
	
	for (int i = 0; i < _size; i++)
		dd += sqrt(_words[i].cn / (double)b);

	for (int i = 0; i < _size; i++) {
		df += sqrt(_words[i].cn / (double)b) / dd;
		if (df > 1) df=1;
		if (df > (a + 1) / (double)class_size) {
			_words[i].class_index = a;
			if (a < class_size - 1) 
				a++;
		} else {
			_words[i].class_index = a;
		}
	}
}

void Vocabulary::print(FILE *fo) {
	for (int a = 0; a < _size; a++) 
		fprintf(fo, "%6d\t%10d\t%s\t%d\n", a, _words[a].cn, _words[a].word, _words[a].class_index);
}

void Vocabulary::scan(FILE *fi) {
	int b;
	
	for (int a = 0; a < _size; a++) {
		fscanf(fi, "%d%d", &b, &_words[a].cn);
		readWord(_words[a].word, fi);
		fscanf(fi, "%d", &_words[a].class_index);
	}
}

void Vocabulary::grow() {
	if (_max_size < _size) {
		if (_words != NULL) 
			free(_words);
		_max_size = _size + 1000;
		_words = (Word *)calloc(_max_size, sizeof(Word));    //initialize memory for vocabulary
	}
}
