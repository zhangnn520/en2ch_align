#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：awesome-align-master 
@File    ：calc_word_vector.py
@Author  ：znn
@Date    ：2022/9/6 14:07 
"""
import os
import torch
import transformers
import itertools

base_dir = os.getcwd()
model_path = os.path.join(base_dir, "checkpoint-30000")
model = transformers.BertModel.from_pretrained(model_path)
tokenizer = transformers.BertTokenizer.from_pretrained(model_path)


def model_eval_fuc(source_string, target_string):
    # alignment
    model.eval()
    align_layer = 8
    # threshold = 1e-3
    threshold = 0.001
    align_words_dict = {}
    align_words_list = []
    sub2word_map_src = []
    sub2word_map_tgt = []
    sent_src, sent_tgt = source_string.strip().split(), target_string.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in
                                                                             sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for
                                                                                 x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                   model_max_length=tokenizer.model_max_length, truncation=True)[
                           'input_ids'], \
                       tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                   truncation=True, model_max_length=tokenizer.model_max_length)[
                           'input_ids']
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)

    for i, j in align_subwords:
        key = sent_src[sub2word_map_src[i]]
        value = sent_tgt[sub2word_map_tgt[j]]
        if [key, value] not in align_words_list:
            align_words_list.append([key, value])

    return align_words_list
