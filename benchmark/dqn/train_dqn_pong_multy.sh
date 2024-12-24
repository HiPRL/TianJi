#!/bin/bash


exp_num=1
finish_reward=21
exp_name="dqn_pong"

old_batch_size=32    # 运行前记得config.py赋值32
old_lr=0.0005
old_senditer=128
batch_size=32
lr=0.0005
node_core=39

for group_prefix in 2
do
    for actor_prefix in 2
    do
        # 创建 hostfile, rankfile
        rm -f tmprankfile$group_prefix
        rm -f tmphostfile$group_prefix
        touch tmprankfile$group_prefix
        touch tmphostfile$group_prefix

        if [ "$group_prefix" = "1" ];then
            host1=cn32
            echo "$host1" >> tmphostfile$group_prefix
            a_num=$(($actor_prefix))
            core_id=$node_core  # actor绑核从最后一个开始
            echo "rank 0=$host1 slot=1-32" >> tmprankfile$group_prefix    # learner
            for rank_id in `seq 1 $a_num`
            do
                echo "rank $rank_id=$host1 slot=$core_id" >> tmprankfile$group_prefix
                core_id=$(($core_id-1))
            done
            echo "rank $(($a_num+1))=$host1 slot=$(($core_id-1))-$core_id" >> tmprankfile$group_prefix    # buffer
            echo "rank $(($a_num+2))=$host1 slot=0" >> tmprankfile$group_prefix
            cat tmprankfile$group_prefix
        elif [ "$group_prefix" = "2" ];then
            host1=cn37
            echo "$host1" >> tmphostfile$group_prefix

            # 组内rank号和分布式一致; root_Learner在最后一个rank.
            learner_cores=16
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

        elif [ "$group_prefix" = "4" ];then
            host1=cn38
            host2=cn39
            echo "$host1" >> tmphostfile$group_prefix
            echo "$host2" >> tmphostfile$group_prefix
            
            learner_cores=16
            a_num=$(($actor_prefix))

            # 1个node有2个group
            for node_id in `seq 1 $(($group_prefix/2))`
            do
                if [ "$node_id" = "1" ];then
                    hostnode=$host1
                elif [ "$node_id" = "2" ];then
                    hostnode=$host2
                fi

                # group 1
                group_id=1
                core_id=$(($learner_cores))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=1-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))

                # group 2
                group_id=2
                start_id=$core_id
                core_id=$(($core_id+$learner_cores-1))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$start_id-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))
            done

            # root_learner
            echo "rank $((($a_num+2)*$group_prefix))=$hostnode slot=0" >> tmprankfile$group_prefix
            cat tmprankfile$group_prefix

        elif [ "$group_prefix" = "6" ];then
            host1=cn32
            host2=cn33
            host3=cn34
            echo "$host1" >> tmphostfile$group_prefix
            echo "$host2" >> tmphostfile$group_prefix
            echo "$host3" >> tmphostfile$group_prefix
            
            learner_cores=16
            a_num=$(($actor_prefix))

            for node_id in `seq 1 $(($group_prefix/2))`
            do
                if [ "$node_id" = "1" ];then
                    hostnode=$host1
                elif [ "$node_id" = "2" ];then
                    hostnode=$host2
                elif [ "$node_id" = "3" ];then
                    hostnode=$host3
                fi

                # group 1
                group_id=1
                core_id=$(($learner_cores))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=1-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))

                # group 2
                group_id=2
                start_id=$core_id
                core_id=$(($core_id+$learner_cores-1))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$start_id-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))

            done

            # root_learner
            echo "rank $((($a_num+2)*$group_prefix))=$hostnode slot=0" >> tmprankfile$group_prefix
            cat tmprankfile$group_prefix
        elif [ "$group_prefix" = "8" ];then
            host1=cn32
            host2=cn33
            host3=cn34
            host4=cn36
            echo "$host1" >> tmphostfile$group_prefix
            echo "$host2" >> tmphostfile$group_prefix
            echo "$host3" >> tmphostfile$group_prefix
            echo "$host4" >> tmphostfile$group_prefix
            
            learner_cores=16
            a_num=$(($actor_prefix))

            for node_id in `seq 1 $(($group_prefix/2))`
            do
                if [ "$node_id" = "1" ];then
                    hostnode=$host1
                elif [ "$node_id" = "2" ];then
                    hostnode=$host2
                elif [ "$node_id" = "3" ];then
                    hostnode=$host3
                elif [ "$node_id" = "4" ];then
                    hostnode=$host4
                fi

                # group 1
                group_id=1
                core_id=$(($learner_cores))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=1-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))

                # group 2
                group_id=2
                start_id=$core_id
                core_id=$(($core_id+$learner_cores-1))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$start_id-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))

            done

            # root_learner
            echo "rank $((($a_num+2)*$group_prefix))=$hostnode slot=0" >> tmprankfile$group_prefix
            cat tmprankfile$group_prefix
        
        elif [ "$group_prefix" = "10" ];then
            host1=cn32
            host2=cn33
            host3=cn34
            host4=cn36
            host5=cn38
            echo "$host1" >> tmphostfile$group_prefix
            echo "$host2" >> tmphostfile$group_prefix
            echo "$host3" >> tmphostfile$group_prefix
            echo "$host4" >> tmphostfile$group_prefix
            echo "$host5" >> tmphostfile$group_prefix
            
            learner_cores=16
            a_num=$(($actor_prefix))

            for node_id in `seq 1 $(($group_prefix/2))`
            do
                if [ "$node_id" = "1" ];then
                    hostnode=$host1
                elif [ "$node_id" = "2" ];then
                    hostnode=$host2
                elif [ "$node_id" = "3" ];then
                    hostnode=$host3
                elif [ "$node_id" = "4" ];then
                    hostnode=$host4
                elif [ "$node_id" = "5" ];then
                    hostnode=$host5
                fi

                # group 1
                group_id=1
                core_id=$(($learner_cores))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=1-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))

                # group 2
                group_id=2
                start_id=$core_id
                core_id=$(($core_id+$learner_cores-1))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$start_id-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))
            done

            # root_learner
            echo "rank $((($a_num+2)*$group_prefix))=$hostnode slot=0" >> tmprankfile$group_prefix
            cat tmprankfile$group_prefix

        elif [ "$group_prefix" = "16" ];then
            host1=cn31
            host2=cn32
            host3=cn33
            host4=cn34
            host5=cn36
            host6=cn37
            host7=cn38
            host8=cn39
            echo "$host1" >> tmphostfile$group_prefix
            echo "$host2" >> tmphostfile$group_prefix
            echo "$host3" >> tmphostfile$group_prefix
            echo "$host4" >> tmphostfile$group_prefix
            echo "$host5" >> tmphostfile$group_prefix
            echo "$host6" >> tmphostfile$group_prefix
            echo "$host7" >> tmphostfile$group_prefix
            echo "$host8" >> tmphostfile$group_prefix
            
            learner_cores=16
            a_num=$(($actor_prefix))

            for node_id in `seq 1 $(($group_prefix/2))`
            do
                if [ "$node_id" = "1" ];then
                    hostnode=$host1
                elif [ "$node_id" = "2" ];then
                    hostnode=$host2
                elif [ "$node_id" = "3" ];then
                    hostnode=$host3
                elif [ "$node_id" = "4" ];then
                    hostnode=$host4
                elif [ "$node_id" = "5" ];then
                    hostnode=$host5
                elif [ "$node_id" = "6" ];then
                    hostnode=$host6
                elif [ "$node_id" = "7" ];then
                    hostnode=$host7
                elif [ "$node_id" = "8" ];then
                    hostnode=$host8
                fi

                # group 1
                group_id=1
                core_id=$(($learner_cores))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=1-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))

                # group 2
                group_id=2
                start_id=$core_id
                core_id=$(($core_id+$learner_cores-1))
                echo "rank $((0+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$start_id-$core_id" >> tmprankfile$group_prefix    # learner
                core_id=$(($core_id+1))
                for rank_id in `seq 1 $a_num`
                do
                    echo "rank $(($rank_id+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix
                    core_id=$(($core_id+1))
                done
                echo "rank $(($a_num+1+($a_num+2)*($group_id-1)+($a_num+2)*2*($node_id-1)))=$hostnode slot=$core_id" >> tmprankfile$group_prefix    # buffer
                core_id=$(($core_id+1))
            done

            # root_learner
            echo "rank $((($a_num+2)*$group_prefix))=$hostnode slot=0" >> tmprankfile$group_prefix
            cat tmprankfile$group_prefix

        fi
        


        # # 修改hyp参数
        # if [ "$actor_prefix" = "8" ];then
        #     batch_size=128
        #     lr=0.0005
        # elif [ "$actor_prefix" = "16" ];then
        #     batch_size=256
        #     lr=0.0005
        # elif [ "$actor_prefix" = "32" ];then
        #     batch_size=256
        #     lr=0.0005
        # fi
        # echo "batch_size $batch_size" "lr $lr"
        # sed -i "s/    batch_size = $old_batch_size/    batch_size = $batch_size/g" benchmark/dqn/dqn_pong_group_distributed.py
        # sed -i "s/    LR = $old_lr,/    LR = $lr,/g" benchmark/dqn/dqn_pong_group_distributed.py
        # old_batch_size=$batch_size
        # old_lr=$lr


        # 运行
        n_t=$((($actor_prefix+2)*$group_prefix+1))
        for _ in `seq 1 $exp_num`
        do  
            mpirun -np $n_t --report-bindings --map-by rankfile:file=tmprankfile$group_prefix --hostfile tmphostfile$group_prefix python benchmark/benchmark_train.py \
            --source benchmark/dqn/dqn_pong_group_distributed.py --exp-name "$actor_prefix$group_prefix$exp_name" --actor-num $actor_prefix --group-num $group_prefix
        done
    done
done

# # 结束后把配置改回默认值
# sed -i "s/    batch_size = $old_batch_size/    batch_size = 32/g" benchmark/dqn/dqn_pong_group_distributed.py
# sed -i "s/    LR = $old_lr,/    LR = 0.0005,/g" benchmark/dqn/dqn_pong_group_distributed.py


# 评估
# python eval_actor.py --exp experiments --exp-name $exp_name --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8,16,32 --exp-suffix "$exp_num,$exp_num,$exp_num,$exp_num,$exp_num,$exp_num"
# python eval_actor.py --exp experiments --exp-name "1$exp_name" --finish-reward $finish_reward --mode multi_actor --exp-prefix 1,2,4,8,16,32 --exp-suffix "$exp_num,$exp_num,$exp_num,$exp_num,$exp_num,$exp_num"
# python eval_actor.py --exp experiments --exp-name "1$exp_name" --finish-reward $finish_reward --mode multi_actor --exp-prefix 8,16,32 --exp-suffix "$exp_num,$exp_num,$exp_num"
# python eval_actor.py --exp experiments --exp-name "dqn_cartpole_si${senditer}" --finish-reward 300 --mode multi_actor --exp-prefix 16, --exp-suffix "3,"
