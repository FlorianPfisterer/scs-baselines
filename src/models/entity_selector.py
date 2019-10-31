import src.models.models as models
import torch
import torch.nn as nn


class EntitySelector(nn.Module):
    """
    Select entities using hidden state from encoder

    Input:
        hidden: hidden state from encoder
        entities: entity vals at start of EntNet cycle
                  (batch_size x num_entities x entity_size)
        keys: key vector for entities
        prev_attn: previous step's attention distribution

    Output:
        selected_entities: selected entity vectors from attention distribution
        entity_dist: attention distribution over entities
        acts: activations of attention distribution
    """
    def __init__(self, opt):
        super(EntitySelector, self).__init__()

        self.initialize_attention(opt)

        # Initialize ppreprocessing projections of
        # sentence encoder hiddens state
        self.preprocess = nn.Sequential()
        for i in range(opt.eNL):
            self.preprocess.add_module(
                "proj_{}".format(i), nn.Linear(opt.hSize, opt.hSize))
            self.preprocess.add_module(
                "act_{}".format(i), nn.PReLU(opt.hSize))
            if opt.edpt != 0 and i < opt.eNL - 1:
                self.preprocess.add_module(
                    "edpt_{}".format(i), nn.Dropout(opt.edpt))

        self.opt = opt

    def initialize_attention(self, opt):
        # Initialize attention function
        # Use bilinear attention
        self.attention = models.BilinearAttention(
            2 * opt.eSize, opt.hSize, 1,
            bias=False, act="sigmoid")

        # If recurrent attention is used,
        # initialize it
        if opt.rec:
            self.choice = nn.Sequential(nn.Linear(opt.hSize, 2),
                                        nn.Softmax(dim=1))
            self.choice_attender = models.Attender()

        self.attender = models.Attender()

    def forward(self, hidden, entities, keys, prev_attn):
        # Transform sentence encoder representation
        preprocessed = self.preprocess(hidden)

        # Compute attention with respect to entity memory
        dist, acts = self.forward_attention(keys, preprocessed, entities)

        # Use recurrent attention
        entity_dist, choice_dist = self.recurrent_attention(
            dist, preprocessed, prev_attn)

        # Attend to entities with final distribution
        selected_entities = self.attender(
            entities, entity_dist, self.opt.eRed)

        return selected_entities, entity_dist, acts

    def forward_attention(self, keys, preprocessed, entities):
        # Attend preprocessed hidden state to entities
        dist, act = self.attention(torch.cat(
            [entities, keys], 2), preprocessed)
        return dist, act

    def recurrent_attention(self, dist, preprocessed, prev_attn):
        batch_size = preprocessed.size(0)
        if self.opt.rec and prev_attn is not None:
            # Compute choice over whether to use current step's attention
            # or fall back to previous step's entity attention
            choice_dist = self.choice(preprocessed)
            return self.choice_attender(torch.cat(
                [prev_attn.view(batch_size, 1, -1),
                 dist.view(batch_size, 1, -1)], 1), choice_dist), choice_dist
        else:
            return dist, None

    def cuda(self, device_id):
        super(EntitySelector, self).cuda(device_id)
        self.attention.cuda(device_id)
        if self.opt.rec:
            self.choice.cuda(device_id)
            self.choice_attender.cuda(device_id)
        self.attender.cuda(device_id)
        self.preprocess.cuda(device_id)
        self.is_cuda = True