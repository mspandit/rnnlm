#!/bin/bash

make clean
make

rm model
rm model.output.txt

time ./rnnlm -train train -valid valid -rnnlm model -hidden 40 -rand-seed 1 -debug 2 -bptt 3 -class 200

#exit

ngram-count -text train -order 5 -lm templm -kndiscount -interpolate -gt3min 1
ngram -lm templm -order 5 -ppl test -debug 2 > temp.ppl

gcc convert.c -o convert
./convert <temp.ppl >srilm.txt

time ./rnnlm -rnnlm model -test test -lm-prob srilm.txt -lambda 0.75
