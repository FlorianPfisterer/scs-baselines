"""
File for methods related to loading configuration files
and setting experiment parameters
author: Antoine Bosselut (atcbosselut)
"""
device = 0


def get_glove_name(opt, type_="tokens", key="pt"):
    emb_type = getattr(opt, key)
    if type_ == "tokens":
        return "data/{}/tokens.{}{}.vocab".format(
            emb_type, emb_type, opt.iSize)
    else:
        return "data/{}/entities.{}{}.vocab".format(
            emb_type, emb_type, opt.eSize)