import torch
from pytorch_transformers import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokens = []
ids = []

i = 0
with open('data/train_nodevtest_sp.en/text_clean', 'r') as f:
	for line in f:
		line = line.strip()
		token = tokenizer.tokenize(line)
		id = tokenizer.encode(line)
		tokens.append(token)
		ids.append(id)
		i += 1
		if i % 10000 == 0:
			print('process %d lines' % i)

with open('data/train_nodevtest_sp.en/token_bert.en', 'w', encoding='utf-8') as wf:
	for token in tokens:
		wf.write(' '.join(token) + '\n')

with open('data/train_nodevtest_sp.en/id_bert.en', 'w', encoding='utf-8') as wf:
	for id in ids:
		wf.write(' '.join(map(str, id)) + '\n')

