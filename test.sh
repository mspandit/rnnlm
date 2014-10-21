#!/bin/bash
rm model*
CC=g++ make -e
./rnnlm -train train -valid valid -rnnlm model -hidden 15 -rand-seed 1 -debug 2 -class 100 -bptt 4 -bptt-block 10 -direct-order 3 -direct 2
make clean
echo "**************************"
echo "OK if no further output..."
echo "**************************"
diff model mastermodel
diff model.output.txt mastermodel.output.txt
