import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_mean_pool
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rlevmatsim.scripts.util import send_reward_request
from tqdm import tqdm
import yaml
from harl.runners import RUNNER_REGISTRY
from harl.envs.ocp.matsim_gnn import MatsimGNN
from harl.envs.ocp.matsim_xml_dataset import MatsimXMLDataset

def train(args):
    with open(args.args_config, "r") as f:
        algo_args = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.env_config, "r") as f:
        env_args = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MatsimXMLDataset(Path(env_args["config_path"]), env_args["num_agents_per_env"])
    env_args["dataset"] = dataset
    env_args["device"] = device

    if (env_args["charge_model_path"] is not None):
        with open(env_args["charge_model_path"], "rb") as f:
            model = torch.load(f)
    else:
        model = MatsimGNN(len(dataset.edge_attr_mapping))

    env_args["dataset"] = dataset
    env_args["charge_model"] = model

    args = vars(args)

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--args_config", type=str, required=True, help="Path to model yaml config file"
    )
    parser.add_argument(
        "--env_config", type=str, required=True, help="Path to environment yaml config file"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ocp",
        help="Environment name.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="marl_ocp", help="Experiment name."
    )
    args = parser.parse_args()
    train(args)
