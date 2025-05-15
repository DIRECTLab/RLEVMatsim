import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from rlevmatsim.classes.matsim_gnn import MatsimGNN
from rlevmatsim.classes.matsim_xml_dataset import MatsimXMLDataset
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rlevmatsim.scripts.util import send_reward_request

# Assume model = YourModel()
# Assume train_loader is defined elsewhere

def train(args):
    model = args.model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.results_dir)

    for epoch in range(args.epochs):
        model.train()
        target = send_reward_request(args.save_dir, args.dataset, args.time_str)
        output = model(args.dataset.graph)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matsim_config_path", type=str, help="Path to the matsim config.xml file")
    parser.add_argument("--results_dir", type=str, default="./results", help="path to directory where this runs results will be saved")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    args.time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    args.results_dir = Path(args.results_dir / args.time_str)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dataset = MatsimXMLDataset(args.matsim_config_path, args.time_str)
    args.model = MatsimGNN(len(args.datatset.edge_attr_mapping))
    train(args)
