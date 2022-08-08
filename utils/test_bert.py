from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
text = 'headache diagnosis'
token = tokenizer.encode(text, add_special_tokens=True)
print(token)