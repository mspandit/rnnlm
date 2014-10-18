#!/bin/bash
rm model*
make clean
CC=g++ make -e
./rnnlm -train train -valid valid -rnnlm model -hidden 15 -rand-seed 1 -debug 2 -class 100 -bptt 4 -bptt-block 10 -direct-order 3 -direct 2
echo "**************************"
echo "OK if no further output..."
echo "**************************"
diff model mastermodel
diff model.output.txt mastermodel.output.txt
