#!/bin/bash

#example script by Martin Karafiat, 2009, karafiat@fit.vutbr.cz
#this will add word timing to mlf file

inmlf=$1
outmlf=$2

mmfdir=/mnt/matylda3/karafiat/AMI/asrcore/exp/ihmtrain07.wb/ml.hlda.mpe
mmfname=$mmfdir/macros,$mmfdir/MMF.stk

scp=/mnt/matylda3/karafiat/AMI/asrcore/sys05/rt05seval/ihmref/lib/flists/rt05seval.ihmref.P1.scp
dict=/mnt/matylda3/karafiat/AMI/asrcore/sys05/rt05seval/ihmref/lib/dicts/50kdict05v0.ihm.pprob.dct

#config=$mmfdir/config.sv
#/homes/kazi/glembek/share/STK/bin/SVite -A -D -V -t 600.0 100.0 1200.0 -P HTK -G HTK -X plp -C $config -S $scp --SOURCEMMF=$mmfname -I "|awk '/^\"/{print; print \"<s>\"; next} /^\\.$/{print \"</s>\"; print; next} {print}' $inmlf" -i $outmlf --SOURCEDICT=$dict --ALLOWXWRDEXP=T --SOURCEHMMLIST=$mmfdir/xwrd.clustered.mlist 

config=$mmfdir/config.hv
HVite -A -D -V -T 1 -t 600.0 100.0 1200.0 -b '<s>' -X plp -C $config -S $scp -H $mmfdir/MMF -i $outmlf -I $inmlf $dict $mmfdir/xwrd.clustered.mlist 
