# TianJi: Highly Parallelized Reinforcement Learning Training with Relaxed Assignment Dependencies

[![Python Version](https://img.shields.io/badge/python-3.6%2F3.7%2F3.8-green)]() [<img src="https://img.shields.io/badge/license-Apache_2.0-blue">]()

📄<a href="http://arxiv.org/abs/2502.20190">arXiv</a>

TianJi is an effective, scalable and highly parallel reinforcement learning training system. TianJi supports building distributed training tasks with simple configuration files, providing users with a universal API to define environments and algorithms, and even allowing users to freely use system components to design distributed training.

## Installation

First, install Python packages using pip:

```
pip install -r requirements.txt
```

Second, install communication package which is in `pkgs` folder:

```
cd pkgs
pip install hiprlcomm-0.0.1-py3-none-any.whl
```

Finally, install the MPI library, one can choose to install `OpenMPI` or `MPICH`. For example, with the Ubuntu system install `OpenMPI`, one can use this command:

```
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

To install more training scenarios, see [Env doc](docs/environment.md).

## Get started

Reinforcement learing algorithms adapted in TianJi are located under `scripts` folder. Using a simple DQN training as an example to demonstrate how to use TianJi, one can use the command line:

```
python scripts/train.py --source config/dqn/cartpole_config.py --exp-name dqn_test
```

The training results are redirected and output under `experiments` folder when one successfully starts training.

## Distributed training

TianJi decoupled computing components on hardware for how reinforcement learning algorithms are executed in a distributed way. 

Take the simple dqn for example:

```
mpirun -np 6 python scripts/train.py --source config/dqn/cartpole_distribution.py --exp-name dqn_dist 
```

The number after `-np` is the process number (N for short) related to the configuration **parallel_parameters**. This number involves a simple calculation: N = learner_num + actor_num + buffer_num.

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

The group number (group_num for short) represents scaling to N computing groups. In each group, it contains multiple roles. Therefore, the process number N is equal to group_num * (learner_num + actor_num + buffer_num) + 1.

## Code Structure

- `config`: Configuration files for algorithms and environments.
- `scripts`: Main entry.
- `drl`: Model、algorithms and policy implemented using system API.
- `env`: An environment implemented with the system API.
- `pkgs`: System dependency packages.
- `utils`: The system function module provides some important components of the system.
- `docs`: Documentation.

## For algorithm developers

- environment, If you want to add a customized environment, Inheritance uses the system `BaseEnv` API, see more [Env doc](docs/environment.md).
- algorithm, If you want to add a customized algorithm, you need to know `Agent`、`Embryo` and `Model` three system API, see more [algorithm doc](docs/algorithm.md).

## Authors and acknowledgment
```
@inproceedings{
  title={Highly Parallelized Reinforcement Learning Training with Relaxed Assignment Dependencies},
  author={Zhouyu He and Peng Qiao and Rongchun Li and Yong Dou and Yusong Tan},
  booktitle={Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## License
[Apache License 2.0](LICENSE)

