# TianJi: Highly Parallelized Reinforcement Learning Training with Relaxed Assignment Dependencies

[![Python Version](https://img.shields.io/badge/python-3.6%2F3.7%2F3.8-green)]() [<img src="https://img.shields.io/badge/license-Apache_2.0-blue">]()

TianJi is an effective, scalable and highly parallel reinforcement learning distributed training system. TianJi supports building distributed training tasks with simple configuration  files, providing users with a universal API to define environments and  algorithms, and even allowing users to freely use system components to  design distributed training.

## Installation

first Install packages using pip:

```
pip install -r requirements.txt
```

second install communication package that in system pkgs dir:

```
cd pkgs
pip install hiprlcomm-0.0.1-py3-none-any.whl
```

finally, install the mpi library, you can choose to install `OpenMPI` or `MPICH`, with the ubuntu system install `OpenMPI`, you can use this command:

```
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

To install more training scenarios, see [Env doc](docs/environment.md).

## Get started

TianJi reinforcement algorithm entry is located under `scripts`, using a simple DQN example to demonstrate how to used TianJi for training.

To start training using the command line:

```
python scripts/train.py --source config/dqn/cartpole_config.py --exp-name dqn_test
```

The training results are generated under `experiments` folder when you successfully start training.

## Distributed training

TianJi decoupled computing components on hardware for how reinforcement  learning algorithms are executed in parallel or distributed. We do this by abstracting some of the task roles to build a data flow diagram, Each role is responsible for a portion of the computation and the roles can be freely defined, distributed, and parallel.

TianJi distributed training relies on computing hardware and is currently only supported on slurm server.If you are on this server, that is great, the system provides some well  implemented distributed training configuration, let's start distributed  training quickly.

Take the simple dqn for example:

```
mpirun -np 6 python scripts/train.py --source config/dqn/cartpole_distribution.py --exp-name dqn_dist 
```

process number N  is related to the configuration **parallel_parameters**, N involves a simple calculation: N = learner_num + actor_num + buffer_num.

In this mode, the learner_num and buffer_num are specified as 1.

If you want to do outward bound on a larger scale, e.g. to increase the number of **learner**,  add the computational Group parameter to the configuration file.

```
    global_cfg = dict(
        use_group_parallel = True,
        group_num = N
    )
```

Training Command:

```
mpirun -np 21 python scripts/train.py --source config/dqn/cartpole_group_distribution.py --exp-name dqn_dist 
```

N group number represents expansion to N computing groups, A group contains multiple role, so process number = group_num * (learner_num + actor_num + buffer_num) + 1.

## Code Structure

- `config`: Configuration files for algorithms and environments.
- `scripts`: Main entry.
- `drl`: Model、algorithms and policy implemented using system API.
- `env`: An environment implemented with the system API.
- `pkgs`: System dependency packages.
- `utils`: The system function module provides some important components of the system.
- `docs`: system documentation.

## For algorithm developers

- environment, If you want to add a customized environment, Inheritance uses the system `BaseEnv` API, see more [Env doc](docs/environment.md).
- algorithm, If you want to add a customized algorithm, you need to know `Agent`、`Embryo` and `Model` three system API, see more [algorithm doc](docs/algorithm.md).

## Authors and acknowledgment
```
@inproceedings{
  tianji,
  title={Highly Parallelized Reinforcement Learning Training with Relaxed Assignment Dependencies},
  author={Zhouyu He and Peng Qiao and Rongchun Li and Yong Dou and Yusong Tan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}，
  url={}
}
```

## License
[Apache License 2.0](LICENSE)

