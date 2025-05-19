import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_mean_pool
from rlevmatsim.classes.matsim_gnn import MatsimGNN
from rlevmatsim.classes.matsim_xml_dataset import MatsimXMLDataset
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rlevmatsim.scripts.util import send_reward_request
from tqdm import tqdm
import yaml
from harl.runners import RUNNER_REGISTRY

def train(args):
    model = args.model.to(args.device)

    with open(args.args_config, "r") as f:
        algo_args = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.env_config, "r") as f:
        env_args = yaml.load(f, Loader=yaml.FullLoader)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.results_dir)

    pbar = tqdm(range(args.epochs))

    with open(Path(args.results_dir, "args.txt"), "w") as f:
        for key, val in args.__dict__.items():
            f.write(f"{key}:{val}\n")

    model.train()
    for epoch in pbar:
        x, edge_index = args.dataset.linegraph.x.to(args.device), args.dataset.linegraph.edge_index.to(args.device) 
        output = model(x, edge_index)
        response = send_reward_request(args.results_dir, args.dataset, args.time_str)
        target = torch.tensor(response[0]).to(args.device)
        optimizer.zero_grad()
        loss = criterion(output, target)
        pbar.set_postfix(loss=loss.item())
        loss.backward()
        writer.add_scalar("Loss", loss.item(), epoch)
        optimizer.step()
        args.dataset.sample_chargers()
    with open(Path(args.results_dir) / "model.pt", "wb") as f:
        torch.save(model, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matsim_config_path", type=str, help="Path to the matsim config.xml file")
    parser.add_argument("--num_agents_per_env", type=int, required=True, help="Number of agents (clusters) for environment for OCP")
    parser.add_argument(
        "--args_config", type=str, required=True, help="Path to model yaml config file"
    )
    parser.add_argument(
        "--env_config", type=str, required=True, help="Path to environment yaml config file"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model to continue training")
    parser.add_argument("--results_dir", type=str, default="./results", help="path to directory where this runs results will be saved")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
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
        default="RLOCP",
        help="Environment name.",
    )
    args = parser.parse_args()
    parser.add_argument(
        "--exp_name", type=str, default="marl_ocp", help="Experiment name."
    )

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    args.time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    args.results_dir = Path(Path(args.results_dir) / args.time_str)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dataset = MatsimXMLDataset(Path(args.matsim_config_path), args.results_dir, args.time_str, args.num_agents_per_env)

    if (args.model_path is not None):
        with open(args.model_path, "rb") as f:
            args.model = torch.load(f)
    else:
        args.model = MatsimGNN(len(args.dataset.edge_attr_mapping))
    train(args)
