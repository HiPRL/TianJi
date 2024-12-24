#!/bin/bash

exp_num=5
finish_reward=300
exp_name="dqn_cartpole"

old_batch_size=32    # 运行前记得config.py赋值32
old_lr=0.0005
old_senditer=128
batch_size=32
lr=0.0005
node_core=39

for group_prefix in 1
do
    for actor_prefix in 16
    do
        # 创建 hostfile, rankfile
        rm -f tmprankfile
        rm -f tmphostfile
        touch tmprankfile
        touch tmphostfile

        if [ "$group_prefix" = "1" ];then
            host1=cn37
            echo "$host1" >> tmphostfile
            a_num=$(($actor_prefix))
            core_id=$node_core  # actor绑核从最后一个开始
            echo "rank 0=$host1 slot=1-5" >> tmprankfile    # learner
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $rank_id=$host1 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1))=$host1 slot=$(($core_id-1))-$core_id" >> tmprankfile    # buffer
            echo "rank $(($a_num+2))=$host1 slot=0" >> tmprankfile
            cat tmprankfile
        elif [ "$group_prefix" = "2" ];then
            # host1=10.107.3.60
            # host2=10.107.3.70
            host1=cn36
            host2=cn37
            echo "$host1" >> tmphostfile
            echo "$host2" >> tmphostfile

            # 组内rank号和分布式一致; root_Learner在最后一个rank.
            a_num=$(($actor_prefix))
            # group 1
            core_id=$node_core  # actor绑核从最后一个开始
            echo "rank 0=$host1 slot=1-5" >> tmprankfile    # learner
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $rank_id=$host1 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1))=$host1 slot=$(($core_id-1))-$core_id" >> tmprankfile    # buffer
            # group 2
            core_id=$node_core
            echo "rank $((0+$a_num+2))=$host2 slot=1-5" >> tmprankfile
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $(($rank_id+$a_num+2))=$host2 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1+$a_num+2))=$host2 slot=$(($core_id-1))-$core_id" >> tmprankfile    
            # root_learner
            echo "rank $(($a_num+2+$a_num+2))=$host2 slot=0" >> tmprankfile
            cat tmprankfile
        elif [ "$group_prefix" = "3" ];then
            host1=cn36
            host2=cn37
            host3=cn38
            # host1=cn51
            # host2=cn52
            # host3=cn53
            echo "$host1" >> tmphostfile
            echo "$host2" >> tmphostfile
            echo "$host3" >> tmphostfile

            a_num=$(($actor_prefix))
            # group 1
            core_id=$node_core  # actor绑核从最后一个开始
            echo "rank 0=$host1 slot=1-5" >> tmprankfile    # learner
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $rank_id=$host1 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1))=$host1 slot=$(($core_id-1))-$core_id" >> tmprankfile    # buffer
            # group 2
            core_id=$node_core
            echo "rank $((0+$a_num+2))=$host2 slot=1-5" >> tmprankfile
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $(($rank_id+$a_num+2))=$host2 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1+$a_num+2))=$host2 slot=$(($core_id-1))-$core_id" >> tmprankfile
            # group 3
            core_id=$node_core
            echo "rank $((0+$a_num+2+$a_num+2))=$host3 slot=1-5" >> tmprankfile
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $(($rank_id+$a_num+2+$a_num+2))=$host3 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1+$a_num+2+$a_num+2))=$host3 slot=$(($core_id-1))-$core_id" >> tmprankfile    
            # root_learner
            echo "rank $(($a_num+2+$a_num+2+$a_num+2))=$host3 slot=0" >> tmprankfile
            cat tmprankfile
        elif [ "$group_prefix" = "4" ];then
            host1=cn36
            host2=cn37
            host3=cn38
            host4=cn39
            # host1=cn51
            # host2=cn52
            # host3=cn53
            echo "$host1" >> tmphostfile
            echo "$host2" >> tmphostfile
            echo "$host3" >> tmphostfile
            echo "$host4" >> tmphostfile

            a_num=$(($actor_prefix))
            # group 1
            core_id=$node_core  
            echo "rank 0=$host1 slot=1-5" >> tmprankfile    # learner
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $rank_id=$host1 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1))=$host1 slot=$(($core_id-1))-$core_id" >> tmprankfile    # buffer
            # group 2
            core_id=$node_core
            echo "rank $((0+$a_num+2))=$host2 slot=1-5" >> tmprankfile
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $(($rank_id+$a_num+2))=$host2 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1+$a_num+2))=$host2 slot=$(($core_id-1))-$core_id" >> tmprankfile
            # group 3
            core_id=$node_core
            echo "rank $((0+$a_num+2+$a_num+2))=$host3 slot=1-5" >> tmprankfile
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $(($rank_id+$a_num+2+$a_num+2))=$host3 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1+$a_num+2+$a_num+2))=$host3 slot=$(($core_id-1))-$core_id" >> tmprankfile    
            # group 4
            core_id=$node_core
            echo "rank $((0+($a_num+2)*($group_prefix-1)))=$host4 slot=1-5" >> tmprankfile
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $(($rank_id+($a_num+2)*($group_prefix-1)))=$host4 slot=$core_id" >> tmprankfile
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1+($a_num+2)*($group_prefix-1)))=$host4 slot=$(($core_id-1))-$core_id" >> tmprankfile
            
            # root_learner
            echo "rank $((($a_num+2)*$group_prefix))=$host4 slot=0" >> tmprankfile
            cat tmprankfile
        fi


        # 修改hyp参数
        if [ "$actor_prefix" = "8" ];then
            batch_size=128
            lr=0.0005
        elif [ "$actor_prefix" = "16" ];then
            batch_size=256
            lr=0.0005
        elif [ "$actor_prefix" = "32" ];then
            batch_size=256
            lr=0.0005
        fi
        echo "batch_size $batch_size" "lr $lr"
        sed -i "s/    batch_size = $old_batch_size/    batch_size = $batch_size/g" benchmark/dqn/dqn_config_group_distributed.py
        sed -i "s/    LR = $old_lr,/    LR = $lr,/g" benchmark/dqn/dqn_config_group_distributed.py
        old_batch_size=$batch_size
        old_lr=$lr


        # 运行
        n_t=$((($actor_prefix+2)*$group_prefix+1))
        for _ in `seq 1 $exp_num`
        do  
            mpirun -np $n_t --report-bindings --map-by rankfile:file=tmprankfile --hostfile tmphostfile python benchmark/benchmark_train.py \
            --source benchmark/dqn/dqn_config_group_distributed.py --exp-name "$actor_prefix$group_prefix$exp_name" --actor-num $actor_prefix --group-num $group_prefix
        done
    done
done

# # 结束后把配置改回默认值
# sed -i "s/    batch_size = $old_batch_size/    batch_size = 32/g" benchmark/dqn/dqn_config_group_distributed.py
# sed -i "s/    LR = $old_lr,/    LR = 0.0005,/g" benchmark/dqn/dqn_config_group_distributed.py


# 评估
# python eval_actor.py --exp experiments --exp-name $exp_name --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8,16,32 --exp-suffix "$exp_num,$exp_num,$exp_num,$exp_num,$exp_num,$exp_num"
# python eval_actor.py --exp experiments --exp-name "1$exp_name" --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8,16,32 --exp-suffix "$exp_num,$exp_num,$exp_num,$exp_num,$exp_num,$exp_num"
# python eval_actor.py --exp experiments --exp-name "1$exp_name" --finish-reward $finish_reward --mode multi_actor --exp-prefix 8,16,32 --exp-suffix "$exp_num,$exp_num,$exp_num"
# python eval_actor.py --exp experiments --exp-name "dqn_cartpole_si${senditer}" --finish-reward 300 --mode multi_actor --exp-prefix 16, --exp-suffix "3,"