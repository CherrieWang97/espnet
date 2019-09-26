import json

read_json_file = "dump/train_nodevtest_sp.de/deltafalse/data_bpe106.lc.json"
write_json_file = "dump/train_nodevtest_sp.de/deltafalse/data_xlm.json"

src_token_file = "data/train_nodevtest_sp.en/token_xlm.en"
trg_token_file = "data/train_nodevtest_sp.de/token_xlm.de"

src_id_file = "data/train_nodevtest_sp.en/id_xlm.en"
trg_id_file = "data/train_nodevtest_sp.de/id_xlm.de"

with open(read_json_file) as f:
	train_json = json.load(f)['utts']

with open(src_token_file) as f:
	src_tokens = f.readlines()

with open(trg_token_file) as f:
	trg_tokens = f.readlines()

with open(src_id_file) as f:
	src_ids = f.readlines()

with open(trg_id_file) as f:
	trg_ids = f.readlines()

i = 0
for k, v in train_json.items():
	trg_token = trg_tokens[i].strip().split()
	src_token = src_tokens[i].strip().split()
	train_json[k]['output'][0]['shape'] = [len(trg_token), 64699]
	train_json[k]['output'][0]['token'] = trg_tokens[i].strip()
	train_json[k]['output'][0]['tokenid'] = trg_ids[i].strip()
	train_json[k]['output'][1]['shape'] = [len(src_token), 64699]
	train_json[k]['output'][1]['token'] = src_tokens[i].strip()
	train_json[k]['output'][1]['tokenid'] = src_ids[i].strip()
	i += 1

with open(write_json_file, 'wb') as f:
	f.write(json.dumps({'utts': train_json},  indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
	
