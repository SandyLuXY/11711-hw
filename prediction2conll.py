import numpy as np
pred_labels = np.load('outputs/scibert_uncase/test_predict_159.npy',allow_pickle=True)
dataset = np.load('data/test_submit_scibert_uncase.npy',allow_pickle=True)
for i in range(len(pred_labels)):
    len1 = len(pred_labels[i])
    len2 = dataset[i][1][-1][-1]
    if len2!=len1:
        print('err')
skipped_ids = np.load('data/test_submit_skip_ids.npy',allow_pickle=True)
pred_labels_final = []
for i in range(len(pred_labels)):
    tmp = []
    tmp_pred = pred_labels[i]
    for k in range(len(dataset[i][1])):
        ind = dataset[i][1][k][0]
        tmp.append(tmp_pred[ind])
    pred_labels_final.append(tmp)
output_str = []
dict_strs = ['O',
    "MethodName",
             "HyperparameterName",
             "HyperparameterValue",
             "MetricName",
             "MetricValue",
             "TaskName",
             "DatasetName"]

with open('raw_txt/test2.txt','r', encoding="utf-8") as f:
    all_str = f.read()
lines = all_str.split('\n\n')

skipped_cnt = 0
for line_num in range(len(lines)):
    if line_num in skipped_ids:
        skipped_cnt+=1
        continue
    line = lines[line_num]
    if(len(line))<1:
        continue
    tmp = [0]
    tmp_splits = [[0,1]]
    cnt = 1
    tmp_cnt = cnt
    token_pair = line.split('\n')
    token_str = []
    token_prev = None
    for i in range(len(token_pair)):
        try:
            token, label = token_pair[i].split('\t')
        except:
            continue
        token_now = dict_strs[pred_labels_final[line_num-skipped_cnt][i+1]]
        token_out = token_now
        if token_now!='O':
            if token_now == token_prev:
                token_out = 'I-' + token_now
            else:
                token_out = 'B-' +token_now
        token_prev = token_now
        token_str.append(token+'\t'+token_out+'\n')
    output_str.append(token_str)
skipped_strings = []
with open('eval_para1.txt','r',encoding='utf-8') as f:
    skipped_strings.append(f.read())
with open('eval_para2.txt','r',encoding='utf-8') as f:
    skipped_strings.append(f.read())
with open('eval2.conll','w',encoding='utf-8') as f:
    cnt = 0
    for i in output_str:
        for k in i:
            f.write(k)
        f.write('\n')
        cnt+=1
        if cnt == skipped_ids[0]:
            f.write(skipped_strings[0]+'\n')
            cnt+=1
        if cnt == skipped_ids[1]:
            f.write(skipped_strings[1]+'\n')
            cnt+=1
