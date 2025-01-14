#!/bin/bash

exp_num=1   
alg_name="dqn"
env_name="pong" # "assault" "boxing"
exp_name="${alg_name}_${env_name}"

hostnode=`hostname` 
node_core=8 # 单机总核数
learner_cores=1 # learner占用核数

finish_reward=12


for group_prefix in 2 
do
    for actor_prefix in 1
    do
        # create hostfile, rankfile
        rm -f tmprankfile$group_prefix
        rm -f tmphostfile$group_prefix
        touch tmprankfile$group_prefix
        touch tmphostfile$group_prefix

        if [ "$group_prefix" = "1" ];then
            host1=$hostnode
            echo "$host1" >> tmphostfile$group_prefix
            a_num=$(($actor_prefix))
            core_id=$node_core  
            echo "rank 0=$host1 slot=1" >> tmprankfile$group_prefix    # learner
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $rank_id=$host1 slot=$core_id" >> tmprankfile$group_prefix
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1))=$host1 slot=$(($core_id-1))-$core_id" >> tmprankfile$group_prefix    # buffer
            echo "rank $(($a_num+2))=$host1 slot=0" >> tmprankfile$group_prefix
            cat tmprankfile$group_prefix
        elif [ "$group_prefix" = "2" ];then
            host1=$hostnode
            echo "$host1" >> tmphostfile$group_prefix
            a_num=$(($actor_prefix))

            # group 1
            echo "rank 0=$host1 slot=1-$learner_cores" >> tmprankfile$group_prefix    # learner
            core_id=$(($learner_cores+1))
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $rank_id=$host1 slot=$core_id" >> tmprankfile$group_prefix
                core_id=$(($core_id+1))
            done
            echo "rank $(($a_num+1))=$host1 slot=$core_id" >> tmprankfile$group_prefix    # buffer
            # group 2
            core_id=$(($core_id+1))
            echo "rank $((0+$a_num+2))=$host1 slot=$core_id-$(($core_id+$learner_cores-1))" >> tmprankfile$group_prefix
            core_id=$(($core_id+$learner_cores))
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $(($rank_id+$a_num+2))=$host1 slot=$core_id" >> tmprankfile$group_prefix
                core_id=$(($core_id+1))
            done
            echo "rank $(($a_num+1+$a_num+2))=$host1 slot=$core_id" >> tmprankfile$group_prefix    
            # root_learner
            echo "rank $(($a_num+2+$a_num+2))=$host1 slot=0" >> tmprankfile$group_prefix
            cat tmprankfile$group_prefix
        fi
        

        # run
        n_t=$((($actor_prefix+2)*$group_prefix+1))
        for _ in `seq 1 $exp_num`
        do  
            mpirun -np $n_t --report-bindings --map-by rankfile:file=tmprankfile$group_prefix --hostfile tmphostfile$group_prefix python scripts/train_group.py \
            --source config/${alg_name}/${env_name}_group_distribution.py --exp-name "$actor_prefix$group_prefix$exp_name" --actor-num $actor_prefix --group-num $group_prefix
        done
    done
done


# evalution
# python eval_actor.py --exp experiments --exp-name $exp_name --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8,16,32 --exp-suffix "$exp_num,$exp_num,$exp_num,$exp_num,$exp_num,$exp_num"
