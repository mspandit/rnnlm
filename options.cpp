#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "types.h"
#include "neuron.h"
#include "synapse.h"
#include "rnnlmlib.h"
#include "options.h"

Options::Options(int argc, char ** argv) {
	_argc = argc;
	_argv = argv;
	
	_train_file[0] = 0;
	_train_mode = -1;
	_train_file_set = -1;
}

int Options::argPos(char *str)
{
	for (int a = 1; a < _argc; a++) if (!strcmp(str, _argv[a])) return a;
    
	return -1;
}

int Options::getTrainFileSet(int debug_mode) {
	if (-1 == _train_file_set)
	{
		(void) getTrainFile(debug_mode);
	} 
	return _train_file_set;
}

int Options::getTrainMode(int debug_mode) {
	if (-1 == _train_mode)
	{
		(void) getTrainFile(debug_mode);
	}
	return _train_mode;
}

char * Options::getTrainFile(int debug_mode) {
	if (0 == _train_file[0]) {
		int i = argPos((char *)"-train");
		if (i > 0) {
			if (i + 1 == _argc) {
				printf("ERROR: training data file not specified!\n");
				exit(1);
			}

			strcpy(_train_file, _argv[i + 1]);

			if (debug_mode > 0)
				printf("train file: %s\n", _train_file);

			if (NULL == fopen(_train_file, "rb")) {
				printf("ERROR: training data file %s not found!\n", _train_file);
				exit(1);
			}

			_train_mode = 1;
			_train_file_set = 1;
			return _train_file;
		} else
			exit(1);
	} else
		return _train_file;
}