import collections
import torch
from transformers import RobertaForMaskedLM
from Constants import *
torch.device('cuda:0')

check_point = torch.load("./testmismatch/pytorch_model.bin")

dicts = collections.OrderedDict()
for k, value in check_point.items():

    # if k == "roberta.embeddings.token_type_embeddings.weight":
    #     value = value.repeat(2, 1)
    #     print(value.size())
    # if k == "roberta.embeddings.word_embeddings.weight":
    #     value = value.repeat(30522, 50265)
    #     print(value.size())
    if k == "roberta.embeddings.position_embeddings.weight":
        value = value.cuda()
        value = value.repeat(512, 514)
        print(value.size())
    dicts[k] = value

torch.save(dicts, "./testmismatch/pytorch_model.bin")
Roberta_for_mlm = RobertaForMaskedLM.from_pretrained(MODEL_PATH)