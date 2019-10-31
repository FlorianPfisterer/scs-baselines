import torch.nn as nn


class BilinearApplicator(nn.Module):
    """
    Transforms selected entity embeddings by action functions

    Initialization Args:
        opt.aSize: size of action function embeddings
        opt.eSize: size of entity embeddings

    Input:
        actions: weighted sum of action functions
        entities: entity embeddings or all entity embeddings

    Output:
        output: encoded sentence embedding

    """
    def __init__(self, opt):
        super(BilinearApplicator, self).__init__()
        self.applicator = nn.Bilinear(opt.aSize, opt.eSize,
                                      opt.eSize, False)

    def forward(self, actions, entities):
        # If entity embeddings haven't been reduced by their attention
        # Apply action function directly to each entity embedding
        if entities.dim() == 3:
            bs = entities.size(0)
            num_ents = entities.size(1)
            h_dim = entities.size(2)
            a_dim = actions.size(1)

            parallel_entities = entities.view(-1, h_dim)
            repeated_actions = actions.view(bs, 1, a_dim).repeat(
                1, num_ents, 1).contiguous().view(-1, a_dim)

            out = self.applicator(repeated_actions, parallel_entities)

            return out.view(entities.size(0),
                            entities.size(1),
                            entities.size(2))
        # apply average action function to average entity embedding
        else:
            return self.applicator(actions, entities)

    def cuda(self, device_id):
        super(BilinearApplicator, self).cuda(device_id)
        self.applicator.cuda(device_id)
        self.is_cuda = True