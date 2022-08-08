import torch
from transformers import AutoTokenizer

if __name__ == '__main__':
    model_name = "FPTAI/vibert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    vocab = tokenizer.get_vocab()
    paragraph = "Tuyển Việt Nam về nước"
    tokens = tokenizer(paragraph)
    print(tokenizer.decode(tokens['input_ids']))