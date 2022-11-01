import numpy as np
import sys
dict_strs = ["MethodName",
"HyperparameterName",
"HyperparameterValue",
"MetricName",
"MetricValue",
"TaskName",
"DatasetName"]
dict_annot = {}
dict_annot['O'] = 0
cnt = 1
for_submission = (sys.argv[2] == 'True')
if for_submission:
    file_name = 'test_submit.txt'
else:
    file_name = 'test.conll'

for i in dict_strs:
    b, intermediate = 'B-'+i, 'I-'+i
    dict_annot[b] = cnt
    dict_annot[intermediate] = cnt
    cnt += 1
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
if sys.argv[1]== 'sci':
    model_name = 'allenai/scibert_scivocab_uncased'
    save_model_name = 'scibert_uncase'
else:
    model_name = 'dslim/bert-base-NER'
    save_model_name = 'nerbert'
file_prefix = 'test'
def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForTokenClassification.from_pretrained(model_name)
layer_name = 'dropout'
for (name, module) in model.named_modules():
    if name == layer_name:
        module.register_forward_hook(hook=hook)
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, device = 0)
features_in_hook=[]
features_out_hook=[]
line_splits = [] # start and end of a token
lines_token = [] # tokens
lines_input_str = [] # input strs
labels_per_line = []
print(file_name)
file_path = 'raw_txt/'+file_name
with open(file_path,'r', encoding="utf-8") as f:
    all_str = f.read()
lines = all_str.split('\n\n')
print(len(lines))

if for_submission:
    split_exp = '\t'
else:
    split_exp = ' '
for line in lines:
    if(len(line))<1:
        continue
    tmp = [0]
    tmp_splits = [[0,1]]
    cnt = 1
    tmp_cnt = cnt
    token_pair = line.split('\n')
    token_str = []
    label_lin = []
    for tokens in token_pair:
        try:
            token, label = tokens.split(split_exp)
        except:
            continue
        if label not in dict_annot:
            print(label, file_name)
            continue
        for i in range(1,len(tokenizer.encode(token))-1):
            tmp.append(dict_annot[label])
            tmp_cnt+=1
        tmp_splits.append([cnt, tmp_cnt])
        cnt = tmp_cnt
        token_str.append(token)
        label_lin.append(dict_annot[label])
    tmp.append(0)

    tmp_splits.append([cnt, cnt+1])
    if(len(tmp)) <3:
        continue
    labels_per_line.append(label_lin)
    lines_token.append(np.array(tmp))
    lines_input_str.append(' '.join(token_str))
    line_splits.append(np.array(tmp_splits))

dataset = []
# 0 feats, 1 start end, 2 token, 3 input text
skipped_ids = []
for i in range(len(lines_input_str)):
    line_tmp = lines_input_str[i]
    try:
        nlp(line_tmp)
    except:
        print(i)
        skipped_ids.append(i)
        features_out_hook.append([0])
        continue
    if len(features_out_hook[i][0]) == 512:
        print(i)
        skipped_ids.append(i)
for i in range(len(lines_input_str)):
    if i in skipped_ids:
        print(i)
        continue
    dataset.append([features_out_hook[i][0].cpu().numpy(), line_splits[i], lines_token[i], lines_input_str[i], labels_per_line[i]])
print(len(dataset))
if for_submission:
    np.save('data/submission_' + save_model_name, dataset, allow_pickle=True)
else:
    np.save('data/test_2_' + save_model_name, dataset, allow_pickle=True)
if for_submission:
    np.save('data/test_submit_skip_ids',skipped_ids, allow_pickle=True)
else:
    np.save('data/test_2_skip_ids',skipped_ids, allow_pickle=True)
if not for_submission:
    np.save('data/test_2_skip_ids',skipped_ids, allow_pickle=True)
