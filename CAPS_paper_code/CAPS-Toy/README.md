# RL Smoothness

RL Smoothness is a Python library for dealing with regularization in Reinforcement learning. This branch contains the toy environment, a simple environment meant to showcase the core issues with current state-of-the-art RL algorithms when it comes to smoothness in actions.

## Installation

### Via pip
It is recommended to set up a python [virtual environment](https://docs.python.org/3/library/venv.html) and enter it, then:
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage


### Train an agent with TD3 on The toy perlin stateful environment with regularization
```bash
python rl_smoothness/train.py td3 --env_type perlin --test_env_type perlin --reg_a --reg_s
```


### Testing a PID controller on the same environment
```bash
python rl_smoothness/test_pid.py
```

## Regularization implementation location
For DDPG, the addition of the regularization losses introduced in this work are between lines 173-179 in the file rl_smoothness/algs/ddpg/ddpg.py


## References

[OpenAI gym](https://gym.openai.com/)

[Spinning up](https://spinningup.openai.com/en/latest/index.html)
