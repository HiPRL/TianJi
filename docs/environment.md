# environment

​	The system now has two examples of encapsulated environments `GymEnv` and `StarCraft2Env` , using  `GymEnv`requires installing gym related libraries, Then it can be used in the configuration of environment parameters, learn more about atari games browse [openai gym](https://github.com/openai/gym). to use `StarCraft2Env`, you need to install smac that installation procedures visit [SMAC docs](https://github.com/oxwhirl/smac/blob/master/README.md).

​	If you want to customize the environment, the system provides a very simple API `BaseEnv`, you just need to implement the interface, contains the following `init step render reset close`  interfaces. Just as you instantiate and call its semantically relevant interface, of course you can also add your own features.