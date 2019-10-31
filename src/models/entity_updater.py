import torch.nn as nn


class EntityUpdater(nn.Module):
    """
    Update entities using entity values changed by applicator

    Initialization Args:
        opt.eSize: size of entities
        opt.afunc: function to use for entity update
                    a = include applicator output (probably should always)
                    k = include projected keys (from REN paper)
                    v = include projected values (from REN paper)
                    c = include projected context (from REN paper)

    Input:
        entities: entity vals at start of NPN cycle
                  (batch_size x num_entities x entity_size)
        new_entities: new_entity from applicator
                  (batch_size x entity_size)
        dist: attention distribution from selecting entities for change
        keys: key vector for entities
        ctx (default=None): context from sentence encoder

    Output:
        updated entity vectors (batch_size x num_entities x entity_size)

    """
    def __init__(self, opt):
        super(EntityUpdater, self).__init__()
        self.opt = opt

        # Update entities with projected context (See EntNet Paper)
        if "c" in self.opt.afunc:
            print "Context Application"
            self.ctx_applicator = nn.Linear(
                opt.hSize, opt.eSize, False)

        self.act = nn.PReLU(opt.eSize)

        # Initialize PReLU negative zone slope to 1
        if "1" in self.opt.act:
            self.act.weight.data.fill_(1)

    def forward(self, entities, new_entities, dist, keys, ctx=None):
        batch_size = entities.size(0)
        num_items = entities.size(1)
        hidden_size = entities.size(2)

        # get update values from context and from applicator
        oc, oa = self.compute_update_components(
            entities, new_entities, keys, ctx)

        # Format attention weights for memory overwrite
        n_dist = dist.view(batch_size, num_items, 1).expand(
            batch_size, num_items, hidden_size)

        # Update the entity memory
        new_ents = self.update_entities(oc, oa, n_dist, entities)

        # Normalize entities
        return new_ents * (1 / new_ents.norm(2, 2)).unsqueeze(2).expand(
            batch_size, num_items, hidden_size), n_dist

    def update_entities(self, oc, oa, n_dist, entities):
        batch_size = entities.size(0)
        num_items = entities.size(1)
        hidden_size = entities.size(2)

        # Just add on contribution of new entities to old ones
        if "n" not in self.opt.afunc:
            # Identity activation
            if self.opt.act == "I":
                new_ents = ((n_dist * (oa + oc)) + (1 - n_dist) *
                            entities)
            # PReLU activation
            else:
                pre_act = (oa + oc).view(-1, hidden_size)
                new_ents = ((n_dist * self.act(pre_act).view(
                    batch_size, num_items, -1)) + (1 - n_dist) * entities)
        # Do interplation of previous entities and current entities
        else:
            if self.opt.act == "I":
                new_ents = entities + n_dist * (oa + oc)
            else:
                pre_act = (oa + oc).view(-1, hidden_size)
                new_ents = (n_dist * self.act(pre_act).view(
                    batch_size, num_items, -1) + entities)

        return new_ents

    def compute_update_components(self, entities, new_entities, keys, ctx):
        batch_size = entities.size(0)
        num_items = entities.size(1)
        hidden_size = entities.size(2)

        oc = 0  # context contribution
        oa = 0  # NPN-changed entity contribtuion

        # NPN contribution
        if new_entities.dim() == 3:
            oa = new_entities
        else:
            oa = new_entities.view(batch_size, 1, hidden_size).expand(
                batch_size, num_items, hidden_size)

        # Context contribution from REN paper
        oc = self.ctx_applicator(ctx).view(batch_size, 1, -1).repeat(
            1, num_items, 1)

        return oc, oa

    def cuda(self, device_id):
        super(EntityUpdater, self).cuda(device_id)
        self.is_cuda = True
        self.act.cuda(device_id)