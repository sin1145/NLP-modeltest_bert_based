import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from transformers import AlbertModel,RobertaModel
from Constants import *

class BertClassifier(nn.Module):

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        # Binary classification problem (num_labels = 2)
        self.num_labels = config.num_labels
        # Pre-trained BERT model
        if MODEL_TYPE == 'albert':
            self.bert = AlbertModel.from_pretrained(MODEL_PATH, config=config)
        if MODEL_TYPE == 'bert':
            self.bert = BertModel.from_pretrained(MODEL_PATH, config=config)
        if MODEL_TYPE == 'roberta':
            self.bert = RobertaModel.from_pretrained(MODEL_PATH, config=config)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # A single layer classifier added on top of BERT to fine tune for binary classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Weight initialization
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        # Forward pass through pre-trained BERT

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # Last layer output (Total 12 layers)
        pooled_output = outputs[-1]

        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)