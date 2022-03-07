# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# Algorithms
from rl_smoothness.algs.ddpg.ddpg import ddpg as ddpg_tf1
from rl_smoothness.algs.ppo.ppo import ppo as ppo_tf1
from rl_smoothness.algs.baseline_ppo.ppo.trainer import train as baseline_ppo_tf1
from rl_smoothness.algs.sac.sac import sac as sac_tf1
from rl_smoothness.algs.td3.td3 import td3 as td3_tf1
from rl_smoothness.algs.trpo.trpo import trpo as trpo_tf1
from rl_smoothness.algs.vpg.vpg import vpg as vpg_tf1

# from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
# from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
# from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
# from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
# from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
# from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch

# Loggers
from rl_smoothness.utils.logx import Logger, EpochLogger

# Version
from rl_smoothness.version import __version__