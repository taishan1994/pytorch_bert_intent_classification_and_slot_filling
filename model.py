import torch.nn as nn
from transformers import BertModel
from transformers import BertForTokenClassification

class BertForIntentClassificationAndSlotFilling(nn.Module):
    def __init__(self, config):
        super(BertForIntentClassificationAndSlotFilling, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_dir)
        self.bert_config = self.bert.config
        self.sequence_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.seq_num_labels),
        )
        self.token_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.token_num_labels),
        )

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                ):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = bert_output[1]
        token_output = bert_output[0]
        seq_output = self.sequence_classification(pooler_output)
        token_output = self.token_classification(token_output)
        return seq_output, token_output


