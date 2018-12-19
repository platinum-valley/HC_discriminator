import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models, datasets
from .dataset.dataset import TuneDataset
from .model.model import Recognizer

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

    parser.add_argument("--input_frame", type=int, default=100,  help="input seaquence length")
    parser.add_argument("--input_dim", type=int, default=13, help="input feature dimention")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="number of encoder layers")
    parser.add_argument("--encoder_hidden_dim", type=int, default=128, help="encoder hidden feature dimention")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="number of decoder layers")
    parser.add_argument("--decoder_hidden_dim", type=int, default=128, help="output feature dimention")
    parser.add_argument("--output_dim", type=int, default=2, help="number of predicted label")
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
    dataset_size = {"train":len(train_dataset), "valid":len(valid_dataset)}
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    recognizer = Recognizer(args)
    optimizer = optim.Adam(recognizer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_model_wts = recognizer.state_dict()
    best_loss = 1e10
    
    criterion = nn.BCELoss(reduction="sum")
    start_time = time.time()

    for epoch in range(args.epoch):
        print("epoch {}".format(epoch+1))

        for phase in ["train", "valid"]:
            if phase == "train":
                recognizer.train(True)
            else:
                recognizer.train(False)
            
            running_loss = 0.0
            for i, data in enumerate(loaders[phase]):
                inputs, label = data
                inputs = Variable(inputs).to(device)
                label = Variable(label).to(device)
                if phase == "train":
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)
                optimizer.zero_grad()
                pred = recognizer(inputs)
                loss = criterion(pred, label)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item
                epoch_loss = running_loss / dataset_size[phase] * args.batch_size
                print("{} loss {:5f}".format(phase, epoch_loss))
                if phase == "valid" and epoch_loss < best_loss :
                    best_model_wts = recognizer.state_dict()
                    best_loss = epoch_loss
    elapsed_time = time.time() - start_time
    print("training complete in {:0}s".format(elapsed_time))
    recognizer.load_state_dict(best_model_wts)
    return recognizer

