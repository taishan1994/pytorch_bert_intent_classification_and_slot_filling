import re
import torch
from transformers import BertTokenizer
from config import Args


class InputExample:
    def __init__(self, set_type, text, seq_label, token_label):
        self.set_type = set_type
        self.text = text
        self.seq_label = seq_label
        self.token_label = token_label


class InputFeature:
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 seq_label_ids,
                 token_label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.seq_label_ids = seq_label_ids
        self.token_label_ids = token_label_ids


class Processor:
    @classmethod
    def get_examples(cls, path, set_type):
        raw_examples = []
        with open(path, 'r') as fp:
            data = eval(fp.read())
        for i, d in enumerate(data):
            text = d['text']
            seq_label = d['intent']
            token_label = d['slots']
            raw_examples.append(
                InputExample(
                    set_type,
                    text,
                    seq_label,
                    token_label
                )
            )
        return raw_examples


def convert_example_to_feature(ex_idx, example, tokenizer, config):
    set_type = example.set_type
    text = example.text
    seq_label = example.seq_label
    token_label = example.token_label

    seq_label_ids = config.seqlabel2id[seq_label]
    token_label_ids = [0] * len(text)
    for k, v in token_label.items():
        # print(k,v, text)
        re_res = re.finditer(v, text)
        for span in re_res:
            entity = span.group()
            start = span.start()
            end = span.end()
            # print(entity, start, end)
            token_label_ids[start] = config.nerlabel2id['B-' + k]
            for i in range(start + 1, end):
                token_label_ids[i] = config.nerlabel2id['I-' + k]
    if len(token_label_ids) >= config.max_len - 2:
        token_label_ids = [0] + token_label_ids[:config.max_len - 2] + [0]
    else:
        token_label_ids = [0] + token_label_ids + [0] + [0] * (config.max_len - len(token_label_ids) - 2)
    # print(token_label_ids)

    text = [i for i in text]
    inputs = tokenizer.encode_plus(
        text=text,
        max_length=config.max_len,
        padding='max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    input_ids =  torch.tensor(inputs['input_ids'], requires_grad=False)
    attention_mask =  torch.tensor(inputs['attention_mask'], requires_grad=False)
    token_type_ids =  torch.tensor(inputs['token_type_ids'], requires_grad=False)
    seq_label_ids  = torch.tensor(seq_label_ids, requires_grad=False)
    token_label_ids = torch.tensor(token_label_ids, requires_grad=False)

    if ex_idx < 3:
        print(f'*** {set_type}_example-{ex_idx} ***')
        print(f'text: {text}')
        print(f'input_ids: {input_ids}')
        print(f'attention_mask: {attention_mask}')
        print(f'token_type_ids: {token_type_ids}')
        print(f'seq_label_ids: {seq_label_ids}')
        print(f'token_label_ids: {token_label_ids}')

    feature = InputFeature(
        input_ids,
        attention_mask,
        token_type_ids,
        seq_label_ids,
        token_label_ids,
    )

    return feature


def get_features(raw_examples, tokenizer, args):
    features = []
    for i, example in enumerate(raw_examples):
        feature = convert_example_to_feature(i, example, tokenizer, args)
        features.append(feature)
    return features


if __name__ == '__main__':
    args = Args()
    raw_examples = Processor.get_examples('./data/test_process.json', 'test')
    tokenizer = BertTokenizer.from_pretrained('../../model_hub/chinese-bert-wwm-ext/')
    features = get_features(raw_examples, tokenizer, args)
