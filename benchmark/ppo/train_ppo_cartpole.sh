#!/bin/bash

exp_num=5
finish_reward=300
exp_name="ppo_cartpole"

run_mode="mpi"
if [ $# -eq 2 ];then
    run_mode=$1
fi

old_buffersize=16

for actor_prefix in 1
do
    for buffer_size in 1 2 4 8 16 32 64 
    do
        # 修改hyp参数
        echo "buffer_size $buffer_size" 
        sed -i "s/    buffer_size = $old_buffersize,/    buffer_size = $buffer_size,/g" benchmark/ppo/ppo_cartpole_distributed_config.py
        old_buffersize=$buffer_size

        n_t=$((2+$actor_prefix))
        for _ in `seq 1 $exp_num`
        do
            mpirun -np $n_t --bind-to core python benchmark/benchmark_train.py --source benchmark/ppo/ppo_cartpole_distributed_config.py --exp-name "$buffer_size$exp_name" --actor-num $actor_prefix
        done
    done
done

python eval_actor.py --exp experiments --exp-name $exp_name --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8,16,32,64 --exp-suffix "$exp_num,$exp_num,$exp_num,$exp_num,$exp_num,$exp_num,$exp_num"
# python eval_actor.py --exp experiments --exp-name $exp_name --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8,16 --exp-suffix "2,2,2,2,2"