import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from rlevmatsim.envs.ocp.matsim_gnn import MatsimGNN
from rlevmatsim.envs.ocp.matsim_xml_dataset import MatsimXMLDataset
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

def train(args):
    model = args.model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.results_dir)

    pbar = tqdm(range(args.epochs))

    with open(Path(args.results_dir, "args.txt"), "w") as f:
        for key, val in args.__dict__.items():
            f.write(f"{key}:{val}\n")

    model.train()
    for epoch in pbar:
        args.dataset.sample_chargers()
        x, edge_index = args.dataset.linegraph.x.to(args.device), args.dataset.linegraph.edge_index.to(args.device) 
        output = model(x, edge_index)
        response = args.dataset.send_reward_request()
        target = torch.tensor(response[0]).to(args.device)
        optimizer.zero_grad()
        loss = criterion(output, target)
        pbar.set_postfix(loss=loss.item())
        loss.backward()
        writer.add_scalar("Loss", loss.item(), epoch)
        optimizer.step()
    with open(Path(args.results_dir) / "model.pt", "wb") as f:
        torch.save(model, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matsim_config_path", type=str, help="Path to the matsim config.xml file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model to continue training")
    parser.add_argument("--results_dir", type=str, default="./results", help="path to directory where this runs results will be saved")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    args.time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    args.results_dir = Path(Path(args.results_dir) / args.time_str)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dataset = MatsimXMLDataset(Path(args.matsim_config_path), args.time_str, args.device)

    if (args.model_path is not None):
        with open(args.model_path, "rb") as f:
            args.model = torch.load(f)
    else:
        args.model = MatsimGNN(len(args.dataset.edge_attr_mapping))

    train(args)
