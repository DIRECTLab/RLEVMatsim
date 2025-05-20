import argparse
import torch
from pathlib import Path
import yaml
from harl.runners import RUNNER_REGISTRY
from harl.envs.ocp.matsim_gnn import MatsimGNN
from harl.envs.ocp.matsim_xml_dataset import MatsimXMLDataset
from tqdm import tqdm
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

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
            model = torch.load(f).to(device)
    else:
        model = MatsimGNN(len(dataset.edge_attr_mapping)).to(device)

    if env_args["charge_model_pretraining_epochs"] is not None:
        pbar = tqdm(range(env_args["charge_model_pretraining_epochs"]), desc="Training Matsim predictor model")
        model.train()
        for _ in pbar:
            x, edge_index = dataset.linegraph.x.to(device), dataset.linegraph.edge_index.to(device) 
            output = model(x, edge_index)
            response = dataset.send_reward_request()
            target = torch.tensor(response[0]).to(args.device)
            dataset.optimizer.zero_grad()
            loss = dataset.criterion(output, target)
            pbar.set_postfix(loss=loss.item())
            loss.backward()
            dataset.optimizer.step()
            args.dataset.sample_chargers()

    env_args["dataset"] = dataset
    env_args["charge_model"] = model

    args = vars(args)

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    with open(Path(runner.run_dir) / "matsim_model.pt", "wb") as f:
        torch.save(model, f)
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
