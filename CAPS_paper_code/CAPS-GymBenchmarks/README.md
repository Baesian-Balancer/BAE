# RL Smoothness

RL Smoothness is a Python library for dealing with regularization in Reinforcement learning

## Installation

### Via pip
It is recommended to set up a python [virtual environment](https://docs.python.org/3/library/venv.html) and enter it, then:
```bash
pip install -r requirements.txt
pip install -e .
```

### Via nix
[Install nix](https://nixos.org/download.html)
Call `nix-shell` which enters a shell with all the necessary dependencies as exactly specified at the time (this may take a while the first time)

## Usage

### Train an agent with DDPG on The Pendulum environment with regularization
```bash
python -m rl_smoothness.run ddpg --lam_a 1 --lam_s 5 --env Pendulum-v0
```


### Running frequency analysis for the pendulum environmnet

```bash
python rl_smoothness/utils/freq_analysis
```

## Regularization implementation location
For DDPG, the addition of the regularization losses introduced in this work are between lines 173-179 in the file rl_smoothness/algs/ddpg/ddpg.py


## References

[OpenAI gym](https://gym.openai.com/)

[Spinning up](https://spinningup.openai.com/en/latest/index.html)
