import numpy as np
pred_labels = np.load('outputs/nerbert_Linear/test_predict_159.npy',allow_pickle=True)
dataset = np.load('data/test_2_nerbert.npy',allow_pickle=True)
for i in range(len(pred_labels)):
    len1 = len(pred_labels[i])
    len2 = dataset[i][1][-1][-1]
    if len2!=len1:
        print('err')
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

with open('raw_txt/test.conll','r', encoding="utf-8") as f:
    all_str = f.read()
lines = all_str.split('\n\n')
skipped_cnt = 0
out_gt = []
for line_num in range(len(lines)):
    line = lines[line_num]
    if(len(line))<1:
        continue
    tmp = [0]
    tmp_splits = [[0,1]]
    cnt = 1
    tmp_cnt = cnt
    token_pair = dataset[line_num][3].split(' ')
    token_str = []
    token_gt = []
    token_prev = None
    token_gt_prev = None
    for i in range(len(token_pair)):
        token = token_pair[i]
        token_now = dict_strs[pred_labels_final[line_num-skipped_cnt][i+1]]
        token_gt_now = dict_strs[dataset[line_num][-1][i]]
        token_gt_out = token_gt_now
        token_out = token_now
        if token_now!='O':
            if token_now == token_prev:
                token_out = 'I-' + token_now
            else:
                token_out = 'B-' +token_now
        if token_gt_now!='O':
            if token_gt_now == token_gt_prev:
                token_gt_out = 'I-' + token_gt_now
            else:
                token_gt_out = 'B-' +token_gt_now
        token_prev = token_now
        token_str.append(token+'\t'+token_out+'\n')
        token_gt.append(token+'\t'+token_gt_out+'\n')
    output_str.append(token_str)
    out_gt.append(token_gt)
with open('test2_ner_linear.conll','w',encoding='utf-8') as f:
    cnt = 0
    for i in output_str:
        for k in i:
            f.write(k)
        f.write('\n')
with open('test2_gt.conll','w',encoding='utf-8') as f:
    cnt = 0
    for i in out_gt:
        for k in i:
            f.write(k)
        f.write('\n')
