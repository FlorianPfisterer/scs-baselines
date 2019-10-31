import torch
import torch.nn as nn
import math


class ActionSelector(nn.Module):
    """
    Compute attention distribution over actions
    Compute average action embedding from this attention distribution

    Initialization Args:
        opt.na: # of actions to initialize
        opt.aSize: size of action embeddings to initialize
        opt.adpt: dropout between MLP layers
        opt.hSize: hidden size of MLP layers
        opt.aNL: number of MLP layers


    Input:
        input: encoded sentence (batch_size x opt.hSize)

    Output:
        average action from selecting over action set

    """
    def __init__(self, opt):
        super(ActionSelector, self).__init__()

        self.selector = nn.Sequential()

        # initialize actions
        self.actions = nn.Parameter(torch.FloatTensor(
            opt.na, opt.aSize))
        stdv = 1. / math.sqrt(self.actions.size(1))
        self.actions.data.uniform_(-stdv, stdv)

        # Build MLP for computing attention distribution over actions
        for i in range(opt.aNL):

            # Add fully connected layer
            self.selector.add_module(
                "proj_{}".format(i), nn.Linear(opt.hSize, opt.hSize))

            # Add activation
            self.selector.add_module(
                "act_{}".format(i), nn.Tanh())

            if opt.adpt != 0:
                # Add dropout layer
                self.selector.add_module(
                    "adpt_{}".format(i), nn.Dropout(opt.adpt))

        # Projection vocabulary
        self.selector.add_module(
            "proj_final", nn.Linear(opt.hSize, opt.na))

        # In actual paper we use a Sigmoid, but since these actions
        # are implicit, I just made it a softmax
        # If we actually ground actions to concepts, we can convert this
        # to a sigmoid and have an L1 normalization step after for the
        # attention distributions
        self.selector.add_module(
            "choose", nn.Softmax(dim=1))

    def forward(self, input):
        return torch.mm(self.selector(input), self.actions)

    def cuda(self, device_id):
        super(ActionSelector, self).cuda(device_id)
        self.selector.cuda(device_id)
        self.actions.cuda(device_id)
        self.is_cuda = True