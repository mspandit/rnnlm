#ifndef _OPTIONS_H_

#define _OPTIONS_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "rnnlmlib.h"

class Options {
private:
	int _argc;
	char ** _argv;
	
	char _train_file[MAX_STRING];
	int _train_mode;
	int _train_file_set;
	
	int argPos(char *);
	
public:
	Options(int, char **);
	
	char * getTrainFile(int);
	int getTrainMode(int);
	int getTrainFileSet(int);
};

#endif 