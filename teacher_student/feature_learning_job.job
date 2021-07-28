#!/bin/sh
#BSUB -q gpu-long
#BSUB -app nvidia-gpu 
#BSUB -env LSB_CONTAINER_IMAGE=ibdgx001:5000/tf_cv:2 
#BSUB -gpu num=2:j_exclusive=no:gmem=2000 
#BSUB -R "select[mem>16000] rusage[mem=16000]" 
#BSUB -o micro_out.%J 
#BSUB -e micro_err.%J
param0=(7 8 9)

maxtry0=3 #${param[@]}
maxtry1=2 #${param[@]}


try0=0
while [ $try0 -lt $maxtry0 ]
do
try1=0
while [ $try1 -lt $maxtry1 ]
do


cat iter.job <(echo python3 micro_feature_learning_integrated.py ${param0[$try0]}  $try1 42 )  > nana.job


sleep .1
done
try1=`expr $try1 + 1`
done
try0=`expr $try0 + 1`
done
