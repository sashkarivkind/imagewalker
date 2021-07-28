#!/bin/sh

if [ "$#" -ne 1 ]; then
    stamp=`date +%s`
else
    stamp=$1
fi


try1=0

param1=(1. 3. 10. 30. 100. 300.)
# 1 5 10 20)
#(0.8 0.9 0.95)
#(0 5 10 12)
#param2=(0 0.1 0.3 0.5 0.7 0.9)
param2=(disabled x 4)
#(0.03 0.1 0.3 1.)
#(0.5 0.8 0.9 0.95 0.99)
maxtry1=5 #${param[@]}
maxtry2=4
while [ $try1 -lt $maxtry1 ]
do
try2=0
while [ $try2 -lt $maxtry2 ]
do


#cat iter.job <(echo python3 run_syclop_generic_cnn_vfb_neu.py 5 ${param1[$try1]}  $stamp)  > nana.job

#bsub  -n 8 -R "select[mem>1024] span[ptile=8] rusage[mem=1024]" <nana.job
#bsub  -R "select[mem>4096] rusage[mem=4000]" <(cat iter.job <(echo python3 run_syclop_generic1.py ${param[$try]} ) )  

bsub -q gpu-medium -n 2 -app nvidia-gpu -env LSB_CONTAINER_IMAGE=ibdgx001:5000/tensorflow1_cv:v3-base-19.12-tf1-py3 -gpu num=2:j_exclusive=no -R "span[ptile=2] select[mem>8192] rusage[mem=8192]" -o out_sweep3.%J -e err_sweep3.%J python run_syclop_generic_cnn_vfb_neu_fade2.py 5 ${param1[$try1]}  $stamp'${LSB_JOBID}_'

try2=`expr $try2 + 1`
sleep .1
done
try1=`expr $try1 + 1`
done
echo stamp for this batch of jobs is $stamp
