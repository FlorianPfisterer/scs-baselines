from models.npn import NPN


class Opt:
    # Entity Selector / Updater
    rec = True      # whether to apply recurrent attention in entity selector
    ents = 100      # number of entity embeddings to initialize
    eSize = 100     # entity embedding dimension
    eNL = 10        # number of MLP layers in entity selector
    edpt = 0.1      # dropout probability in entity selector
    afunc = 'a'     # function to use for entity update (a, k, v, c)
    act = 'I'       # type of entity activation to use (I: identity)

    # Action Selector
    na = 100        # number of action embeddings to initialize
    aSize = 100     # action embedding dimension
    adpt = 0.1      # dropout between MLP layers in action selector
    aNL = 10        # number of MLP layers in action selector

    # Sentence Encoder
    vSize = 100     # number of word embeddings for sentence encoder
    dpt = 0.3       # dropout probability after the embedding layer

    # Shared
    hSize = 100     # dimension of hidden MLP layers / word embedding dimension

def main():
    print('initializing...')
    model = NPN(opt=Opt(), max_words=10)
    print(model)


main()