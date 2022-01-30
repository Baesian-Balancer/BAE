import torch
import torchvision
from omegaconf import OmegaConf
import hydra
import sys 
import os
sys.path.append(os.path.abspath("../"))
from keith_scratch_mbrl.PolicyNet import PolicyNetwork
from agent.sac import SACAgent
from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor
def main():

    ######### Options
    # Load and save paths
    model_file = "/home/capstone/capstone/rl-algorithm-exploration/sac/best_model.pt" # path to .pt file
    ts_model_file = "/home/capstone/capstone/rl-algorithm-exploration/sac/exported_SAC.pt" # save path for output torchscript file

    fixed_hip = False
    export_type = "tracing" # tracing/scripting
    ######### End Options

    
    # # # Load SAC Model
    OBS_SIZE = 10 # Free hip
    ACTION_SIZE = 2
    DEVICE = "cpu"

    agent_cfg = OmegaConf.load('config/agent/sac.yaml')

    agent_cfg.agent.params.device = DEVICE
    agent_cfg.agent.params.batch_size = 0
    agent_cfg.agent.params.obs_dim = OBS_SIZE # self.env.observation_space.shape[0]
    agent_cfg.agent.params.action_dim = ACTION_SIZE
    agent_cfg.agent.params.action_range = [
        float(-1),
        float(1)
    ]
    agent = hydra.utils.instantiate(agent_cfg.agent)
    actor = agent.actor
    # # # agent_cfg.agent.params.critic_cfg = c_cfg
    # # agent_cfg = OmegaConf.to_container(agent_cfg, resolve=True)
    
    # # hidden_dim = agent_cfg["agent"]["params"]["critic_cfg"]["params"]["hidden_dim"]
    # # hidden_depth = agent_cfg["agent"]["params"]["critic_cfg"]["params"]["hidden_depth"]
    # # log_std_bounds = agent_cfg["agent"]["params"]["actor_cfg"]["params"]["log_std_bounds"]

    # # critic = DoubleQCritic(OBS_SIZE, ACTION_SIZE, hidden_dim, hidden_depth)
    # # actor = DiagGaussianActor(OBS_SIZE, ACTION_SIZE, hidden_dim, hidden_depth, log_std_bounds)
    
    # # c_cfg = {"critic": critic, "actor": actor}
    # # agent_cfg["agent"]["params"]["critic_cfg"] = c_cfg

    # # agent = SACAgent(**agent_cfg["agent"]["params"])

    checkpoint = torch.load(model_file)
    
    # # agent.critic = critic
    # # agent.actor = actor
    # # agent.critic.load_state_dict(checkpoint['critic_state_dict'])

    actor.load_state_dict(checkpoint['actor_state_dict'])

    if export_type == "tracing":
        example_obs = torch.rand(OBS_SIZE)
        ts_model = torch.jit.trace(actor, example_obs)
    elif export_type == "scripting":
        ts_model = torch.jit.script(actor)
    
    ts_model.save(ts_model_file)


if __name__ == "__main__":
    main()
