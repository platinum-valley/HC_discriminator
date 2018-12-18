import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from dataset import TuneDataset
from model import Recognizer

import numpy as np
import pickle
import sys
import argparse

def get_argument():
    parser = argparse.ArgumentParser(description="Parameter fro training of network")
    parser.add_argument("--batch_size", type=int, default=64, help="input minibatch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epoch")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate of optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="dropout_rate")
    parser.add_argument("--gpu", action="store_true", help="flag of using gpu")
    parser.add_argument("--train_data", type=str, help="csv file discripted training dataset")
    parser.add_argument("--valid_data", type=str, help="csv file discripted validation dataset")
    args = parser.parse_args()
    return args

def train_recognizer(args):
    if not (args.train_data and args.valid_data):
        print("must set train_data and valid_data")
        sys.exit()
    trans = trainsforms.ToTensor()
    train_dataset = TuneDataset(args.train_data, transform=trans)
    valid_dataset = TuneDataset(args.valid_data, transform=trans)
    train_loader = data_util.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = data_util.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    loaders = {"train":train_loader, "valid":valid_loader}
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    recognizer = Recognizer(args)
