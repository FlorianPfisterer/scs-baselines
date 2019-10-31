import src.models.models as models
import torch.nn as nn


class SentenceEncoder(nn.Module):
    """
    Wrapper class for a sentence encoder

    Initialization Args:
        opt: options to pass to sentence encoder class
        max_words: maximum number of words in any sentence in
                   data

    Input:
        input: word indices for a sentence (batch_size x num_words)
        lengths: number of non-pad indices in each sentence in the batch

    Output:
        output: encoded sentence embedding

    """
    def __init__(self, opt, max_words=None):
        super(SentenceEncoder, self).__init__()
        self.opt = opt
        self.sentence_encoder = models.WeightedBOW(opt, max_words)

    def forward(self, input, lengths):
        output, hidden = self.sentence_encoder(
            input, lengths)

        return output

    def cuda(self, device_id):
        super(SentenceEncoder, self).cuda(device_id)
        self.sentence_encoder.cuda(device_id)
        self.is_cuda = True
