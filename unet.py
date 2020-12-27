import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict,deque


class ContractBlock(nn.Module):

    def __init__(self, in_chans, out_chans, k=3, s=1, p=1):
        super(ContractBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, k, s, p)  # padded
        self.conv2 = nn.Conv2d(out_chans, out_chans, k, s, p)  # padded

    def forward(self, x):
        out = F.relu(self.conv1(x))
        concat = F.relu(self.conv2(out))
        out = F.max_pool2d(concat, 2)  # downsampling
        return out, concat


class ExpandBlock(nn.Module):

    def __init__(self, in_chans, out_chans, k=3, s=1, p=1):
        super(ExpandBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_chans, out_chans, 2, 2)
        self.conv1 = nn.Conv2d(2 * out_chans, out_chans, k, s, p)
        self.conv2 = nn.Conv2d(out_chans, out_chans, k, s, p)

    def forward(self, x, concat):
        out = self.upconv(x)
        out = torch.cat([concat, out], dim=1)  # concatenate on the channel axis
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNet(nn.Module):

    ContractBlock = ContractBlock
    ExpandBlock = ExpandBlock

    def __init__(self, progression_order, n_class=2):
        super(UNet, self).__init__()
        self.progression_order = progression_order
        self.depth = len(self.progression_order)
        self.contraction = nn.Sequential(OrderedDict([
            (
                f"ContractBlock_{i+1}", 
                self.ContractBlock(self.progression_order[i], self.progression_order[i+1])
            ) 
            for i in range(self.depth - 2)
        ]))
        self.expansion = nn.Sequential(OrderedDict([
            (
                f"ExpandBlock_{i+1}", 
                self.ExpandBlock(self.progression_order[i], self.progression_order[i - 1])
            )
            for i in range(self.depth - 1, 1, -1)
        ]))
        self.bottom = nn.Sequential(
            nn.Conv2d(self.progression_order[-2], self.progression_order[-1], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.progression_order[-1], self.progression_order[-1], 3, 1, 1),
            nn.ReLU()
        )
        self.conv1x1 = nn.Conv2d(self.progression_order[1], n_class, 1)

    def forward(self, x):
        concats = deque(maxlen=len(self.contraction))
        for contract_block in self.contraction.children():
            x, concat = contract_block(x)
            concats.append(concat)
        x = self.bottom(x)
        for expand_block in self.expansion.children():
            x = expand_block(x, concats.pop())
        x = self.conv1x1(x)
        return x


def train(model,trainloader,loss_fn,optim,testloader=None,epochs=20,device=device):
    for e in range(epochs):
        running_loss = 0.0
        running_correct = 0.0
        running_pixels = 0
        with tqdm(trainloader,desc=f"{e+1}/{epochs} epochs:") as t:
            for i,batch in enumerate(t):
                x = batch[0].to(device)
                y = batch[1].to(device).squeeze(dim=1)
                out = model(x)
                loss = loss_fn(out,y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running_loss += loss.item()*np.prod(y.shape)
                running_correct += (out.max(dim=1)[1]==y).sum().item()
                running_pixels += np.prod(y.shape)
                if i != len(t)-1:
                    t.set_postfix({
                        "train_loss": running_loss/running_pixels,
                        "train_acc": running_correct/running_pixels
                    })
                else:
                    if testloader is not None:
                        test_running_loss = 0.0
                        test_running_correct = 0.0
                        test_running_pixels = 0
                        with torch.no_grad():
                            for test_batch in testloader:
                                x = test_batch[0].to(device)
                                y = test_batch[1].to(device).squeeze(dim=1)
                                out = model(x)
                                loss = loss_fn(out,y)
                                test_running_loss += loss.item()*np.prod(y.shape)
                                test_running_correct += (out.max(dim=1)[1]==y).sum().item()
                                test_running_pixels += np.prod(y.shape)
                        t.set_postfix({
                            "train_loss": running_loss/running_pixels,
                            "train_acc": running_correct/running_pixels,
                            "test_loss": test_running_loss/test_running_pixels,
                            "test_acc": test_running_correct/test_running_pixels
                        })
                    else:
                        t.set_postfix({
                            "train_loss": running_loss/running_pixels,
                            "train_acc": running_correct/running_pixels
                        })


if __name__ == "__main__":
    
    from utils.data import Datasets
    from torch.optim import Adam
    from loss import FocalLoss

    root = "./dataset"
    dataset = Datasets(root)
    trainloader = DataLoader(dataset.trainset,batch_size=8,shuffle=True)
    testloader = DataLoader(dataset.testset, batch_size=16,shuffle=False)

    progression_order = [3, 64, 128, 256, 512, 1024]
    model = UNet(progression_order)
    # loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss_fn = FocalLoss(lbd=2)
    optim = Adam(model.parameters())
    model.to(device)
    train(model,trainloader,loss_fn,optim,testloader,epochs=1)