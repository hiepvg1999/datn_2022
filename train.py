import argparse
import glob
import logging
import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torchvision
from dataset.dataset import Receipt
from model import BERTxGCN, BERTxSAGE,BERTxGAT
from utils import LABELS
from utils.metrics import MetricTracker
from loss import FocalLoss
parser = argparse.ArgumentParser(description='PyTorch BERT-GCN')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--val-batch-size', type=int, default=4, metavar='N',
                    help='input batch size for validation (default: 4)')
parser.add_argument('--train-folder', type=str,
                    default="dataset/train_data/",
                    help='training folder path')
parser.add_argument('--val-folder', type=str,
                    default="dataset/val_data/",
                    help='validation folder path')
parser.add_argument('--log-dir', type=str,
                    default="logs/runs/",
                    help='TensorBoard folder path')         
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--num-warmup-steps', type=float, default=1000, metavar='N',
                    help='numbers warmup steps (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many epoches to wait before saving model')
parser.add_argument('--save-folder', type=str, default="logs/saved", metavar='N',
                    help='how many epoches to wait before saving model')
parser.add_argument('--pretrain', type= str, default="logs/saved/model_best.pth",
                    help= 'Please enter pretrain')
parser.add_argument('--model_type', type=str, default='SAGE', help='Please enter model type')

def train(epoch):
    model.train()
    train_loss = 0.

    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        for batch_idx, data in enumerate(train_bar):
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            train_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    train_loss /= len(train_loader)
    if log_writer:
        log_writer.add_scalar('train/loss', train_loss, epoch)
    return train_loss

def val(epoch):
    model.eval()
    val_loss = 0.
    for data in tqdm(val_loader, desc="Validation"):
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            output = model(data)
            # sum up batch loss
            val_loss += criterion(output, data.y).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            metric.update(pred, data.y.data.view_as(pred))

    val_loss /= len(val_loader)

    print(f"Classification Report:")
    print(metric.compute())
    if log_writer:
        log_writer.add_scalar('val/loss', val_loss, epoch)
    metric.reset()
    return val_loss


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(args.seed)
    train_files = glob.glob(args.train_folder + "*.json")
    train_dataset = Receipt(train_files)
    print(f"Number of training set: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    val_files = glob.glob(args.val_folder + "*.json")
    val_dataset = Receipt(val_files)
    print(f"Number of validation set: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=0)
    # model = BERTxSAGE()
    if args.model_type.lower() == 'gcn':
        model = BERTxGCN()
        print(repr(model))
    elif args.model_type.lower() == 'gat':
        model = BERTxGAT()
        print(repr(model))
    else:
        model = BERTxSAGE()
        print(repr(model))

    # print("Number of parameters:{}-------------------".format(sum([param.nelement() for param in model.conv1.parameters()]))) 
    # model = torch.load(args.pretrain)
    for param in model.BERT.parameters():
        param.requires_grad = False
    if args.cuda:
        # Move model to GPU.
        model.cuda()

    # criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    # weighted cross entropy loss. reduce 'others' class to 20%
    class_weights = torch.FloatTensor([2.0, 1.5, 1.0, 1.0,1.0, 0.2]).cuda()
    # class_weights = torch.FloatTensor([0.5, 0.75, 2., 1.5]).cuda()  # funsd dataset
    # class_weights = torch.FloatTensor([1.0, 1.0, 1.0,1.0, 1.0]).cuda()  # sroie dataset
    # criterion = torch.nn.NLLLoss(weight=class_weights)
    criterion = FocalLoss(gamma= 0.5)
    # Define optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    t_total = len(train_loader) * args.epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total)
    metric = MetricTracker(labels=LABELS)
    best_loss = -1
    log_writer = SummaryWriter(args.log_dir)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = val(epoch)
        print('Train Epoch: {} \tTrain Loss: {:.6f} \tValidation Loss: {:.6f} \tLearning rate: {}'.format(
            epoch, train_loss, val_loss, lr_scheduler.get_last_lr()))
        if args.save_interval > 0:
            if val_loss < best_loss or best_loss < 0:
                best_loss = val_loss
                print(f"Saving best model, loss: {best_loss}")
                torch.save(model, os.path.join(
                    args.save_folder, "model_focal05_{}.pth".format(datetime.now().strftime("%Y-%m-%d"))))
                continue
            if epoch % args.save_interval == 0:
                print(f"Saving at epoch: {epoch}")
                torch.save(model, os.path.join(
                    args.save_folder, f"model_{epoch}.pth"))
