#!/bin/bash

exp_num=5
finish_reward=20
exp_name="qmix_8m"


for actor_prefix in 1 2 4 8
do
    n_t=$((2+$actor_prefix))

    for _ in `seq 1 $exp_num`
    do
        mpirun -np $n_t --report-bindings --map-by rankfile:file=tmprankfile$actor_prefix --hostfile tmphostfile$actor_prefix python train.py --source benchmark/qmix/qmix_config_distributed$actor_prefix.py --exp-name "$actor_prefix$exp_name"
    done
done

python eval_actor.py --exp experiments --exp-name $exp_name --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8 --exp-suffix "$exp_num,$exp_num,$exp_num,$exp_num"