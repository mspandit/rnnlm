#!/bin/bash
rm model*
make clean
CC=g++ make -e
./rnnlm -train train -valid valid -rnnlm model
echo "**************************"
echo "OK if no further output..."
echo "**************************"
diff model mastermodel
diff model.output.txt mastermodel.output.txt
