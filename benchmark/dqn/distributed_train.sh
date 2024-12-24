#!/bin/bash

node_num=3
run_mode="run_mode"
case $# in
    0)
    ;;
    1)
    node_num=$1
    ;;
    2)
    node_num=$1
    run_mode=$2
    ;;
    *)
    echo "参数个数不超过2个，第一个参数为进程数量（可选），第二个参数为CPU单核模式（可选：s或single）。"
    exit
    ;;
esac

if [ $node_num -lt 3 ];then
    echo "进程数量至少3个"
    exit
fi

if [ $run_mode = "s" ] || [ $run_mode = "single" ];then
    taskset -c 0 mpirun -np $node_num python train.py \
        --source "./config/dqn_config_distributed.py" \
        --exp-name "dist_local_dqn"
else
    mpirun -np $node_num python train.py \
        --source "./config/dqn_config_distributed.py" \
        --exp-name "dist_local_dqn"
fi