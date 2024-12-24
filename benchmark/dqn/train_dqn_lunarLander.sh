#!/bin/bash

exp_num=3
finish_reward=200
exp_name="dqn_lunarLander"

run_mode="mpi"
if [ $# -eq 2 ];then
    run_mode=$1
fi

for actor_prefix in 1 2 4 8
do
    n_t=$((2+$actor_prefix))

    for _ in `seq 1 $exp_num`
    do
        if [ $run_mode = "s" ] || [ $run_mode = "slurm" ];then
            srun -w $2 -n $n_t --cpu-bind=threads python benchmark/benchmark_train.py --source benchmark/dqn/dqn_lunarLander_distributed.py --exp-name "$actor_prefix$exp_name" --actor-num $actor_prefix
        else
            mpirun -np $n_t --bind-to core python benchmark/benchmark_train.py --source benchmark/dqn/dqn_lunarLander_distributed.py --exp-name "$actor_prefix$exp_name" --actor-num $actor_prefix
        fi
    done
done

python eval_actor.py --exp experiments --exp-name $exp_name --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8 --exp-suffix "$exp_num,$exp_num,$exp_num,$exp_num"