# TianJi: Highly Parallelized Reinforcement Learning Training with Relaxed Assignment Dependencies

[![Python Version](https://img.shields.io/badge/python-3.6%2F3.7%2F3.8-green)]() [<img src="https://img.shields.io/badge/license-Apache_2.0-blue">]()

TianJi is an effective, scalable and highly parallel reinforcement learning distributed training system. TianJi supports building distributed training tasks with simple configuration  files, providing users with a universal API to define environments and  algorithms, and even allowing users to freely use system components to  design distributed training.

## Installation

first Install packages using pip：

```
pip install -r requirements.txt
```

second install communication package that in system pkgs dir：

```
pip install yhcomm-0.0.1-py3-none-any.whl
```

Finally, install the mpi library，Note that `MPICH` and `OpenMPI` correspond to `mpi4py` versions。

## Get started

TianJi reinforcement algorithm entry is located under `scripts`, using a simple DQN example to demonstrate how to used TianJi for training.

The first way is to run it directly using a script file:

```
cd tiranji/scripts
bash simple_run.sh
```

The second way is to start training using the command line:

```
python scripts/train.py --source config/dqn/dqn_config.py --exp-name dqn_test
```

The training results are generated under `experiments` folder.

## Distributed training

TianJi decoupled computing components on hardware for how reinforcement  learning algorithms are executed in parallel or distributed. We do this by abstracting some of the task roles to build a data flow diagram, Each role is responsible for a portion of the computation and the roles can be freely defined, distributed, and parallel.

TianJi distributed training relies on computing hardware and is currently only supported on slurm server.If you are on this server, that is great, the system provides some well  implemented distributed training configuration, let's start distributed  training quickly.

Take the simple dqn for example:

```
python scripts/train.py --source config/dqn/dqn_config_distributed.py --exp-name dqn_dist 
```

If you want to do outward bound on a larger scale, add the Group parameter to the configuration file.

```
    global_cfg = dict(
        use_group_parallel = True,
        group_num = N
    )
```

N group number represents expansion to N computing groups, A group contains multiple roles.

## Code Structure

- `benchmark`: Some experimental configurations of benchmark.
- `config`: Configuration files for algorithms and environments.
- `scripts`: Main entry.
- `drl`: Model、algorithms and policy implemented using system API.
- `env`: An environment implemented with the system API.
- `pkgs`: System dependency packages.
- `utils`: The system function module provides some important components of the system.

## For algorithm developers

- environment, If you want to add a customized environment, Inheritance uses the system `BaseEnv` API.
- algorithm, If you want to add a customized algorithm, you need to know `Agent`、`Embryo` and `Model` three system API.

## Authors and acknowledgment
AAAI2025.

## License
[Apache License 2.0]()
