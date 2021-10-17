from typing import Dict
from layers import DConv2d, DLinear, DModule
from logger import Logger
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from hyper_parameters import HyperParameters
from custom_types import IStateDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(DModule):
    def __init__(self):
        super().__init__()

        def conv_with_relu(*args, **kargs):
            return nn.Sequential(
                DConv2d(*args, **kargs),
                nn.ReLU(),
            )

        self.max_pool = nn.MaxPool2d(2, 2)
        self.last_max_pool = nn.MaxPool2d(4, 4)
        self.flatten = nn.Flatten()

        self.cnn_block1 = nn.Sequential(
            conv_with_relu(3, 64, 3, padding=1),
            conv_with_relu(64, 128, 3, padding=1),
        )

        self.cnn_block2 = nn.Sequential(
            conv_with_relu(128, 128, 3, padding=1),
            conv_with_relu(128, 128, 3, padding=1),
        )

        self.cnn_block3 = conv_with_relu(128, 256, 3, padding=1)

        self.cnn_block4 = conv_with_relu(256, 512, 3, padding=1)

        self.cnn_block5 = nn.Sequential(
            conv_with_relu(512, 512, 3, padding=1),
            conv_with_relu(512, 512, 3, padding=1),
        )

        # self.last_max_pooling = nn.MaxPool2d(4, 1)
        self.fcl = DLinear(512, 10)

        self.to(device)

    def forward(self, X):
        out = self.cnn_block1(X)
        out = self.max_pool(out)

        out = self.cnn_block2(out) + out
        out = self.cnn_block3(out)
        out = self.max_pool(out)

        out = self.cnn_block4(out)
        out = self.max_pool(out)

        out = self.cnn_block5(out) + out
        out = self.last_max_pool(out)

        out = self.fcl(self.flatten(out))

        return out
# class Model(DModule):
#     def __init__(self):
#         super().__init__()

#         def conv_with_relu(*args, **kargs):
#             return nn.Sequential(
#                 DConv2d(*args, **kargs),
#                 nn.ReLU(),
#             )

#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.last_max_pool = nn.MaxPool2d(4, 4)
#         self.flatten = nn.Flatten()

#         self.cnn_block1 = nn.Sequential(
#             conv_with_relu(3, 32, 3, padding=1),
#             conv_with_relu(32, 64, 3, padding=1),
#         )

#         self.cnn_block2 = nn.Sequential(
#             conv_with_relu(64, 64, 3, padding=1),
#             conv_with_relu(64, 64, 3, padding=1),
#         )

#         self.cnn_block3 = conv_with_relu(64, 128, 3, padding=1)

#         self.cnn_block4 = conv_with_relu(128, 256, 3, padding=1)

#         self.cnn_block5 = nn.Sequential(
#             conv_with_relu(256, 256, 3, padding=1),
#             conv_with_relu(256, 256, 3, padding=1),
#         )

#         self.last_max_pooling = nn.MaxPool2d(4, 1)
#         self.fcl = DLinear(256, 10)

#         self.to(device)

#     def forward(self, X):
#         out = self.cnn_block1(X)
#         out = self.max_pool(out)

#         out = self.cnn_block2(out) + out
#         out = self.cnn_block3(out)
#         out = self.max_pool(out)

#         out = self.cnn_block4(out)
#         out = self.max_pool(out)

#         out = self.cnn_block5(out) + out
#         out = self.last_max_pool(out)

#         out = self.fcl(self.flatten(out))

#         return out


class Backbone(Model):
    # def __init__(self, dataloader: Dict[str, DataLoader] = None, hyper_parameters: HyperParameters = None):
    #     super().__init__()
    #     self.dataloader = dataloader
    #     self.hyper_parameters = hyper_parameters

    def __init__(self, device: torch.device,
                hyper_parameters: HyperParameters,
                dataloader: Dict[str, DataLoader] = None,
                test_dataloader: Dict[str, DataLoader] = None,
                sigma: IStateDict = None, phi: IStateDict = None,
                client_id: int = None):
        super().__init__()
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.hyper_parameters = hyper_parameters
        self.client_id = client_id #server : client_id = None
        self.logger = Logger(client_id = client_id, device = device)
        self.device = device


    def evaluate(self, train = False):
        '''
        evaluate on training set if train = True.Otherwise on test set/
        '''

        dataloader = self.dataloader['labeled'] if train else self.test_dataloader #if train=True, use 'labeled' only or both 'labeled' and 'unlabeled'?

        correct = 0
        running_loss = 0.
        loss_fn = torch.nn.CrossEntropyLoss()
        self.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)

                pred = self.forward(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                running_loss += loss_fn(pred, y).item() * X.size(0)
        test_acc = correct / len(dataloader.dataset)
        test_loss = running_loss / len(dataloader.dataset)
        return test_acc, test_loss

