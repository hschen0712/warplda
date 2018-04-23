#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 hschen0712 <hschen0712@gmail.com>
#
# Distributed under terms of the MIT license.

"""
读取warplda模型
"""
from collections import defaultdict
import numpy as np

class Warplda(object):
    def __init__(self, model_path, vocab_path):
        with open(model_path, 'rb') as fr:
            lines = [line.strip() for line in fr.readlines()]

        params = lines[0].split()
        self.vocab_size = int(params[0]) #词表大小
        self.num_topics = int(params[1]) #主题数
        self.alpha = float(params[2])
        self.beta = float(params[3])
        self.alpha_bar = self.alpha * self.num_topics
        self.beta_bar = self.beta * self.vocab_size
        self.Cvk = np.zeros((self.vocab_size, self.num_topics))

        
        for word_id, line in enumerate(lines[1:]):
            num_elements, word_topic_cnt = line.split('\t')
            word_id = int(word_id)
            pairs = word_topic_cnt.split()
            for pair in pairs:
                topic_id, cnt = pair.split(':')
                topic_id = int(topic_id)
                cnt = int(cnt)
                self.Cvk[word_id, topic_id] = cnt
        self.Ck = self.Cvk.sum(axis=0)
        # 词表加载
        self.vocab_map = {}
        with open(vocab_path) as fr:
            lines = [line.strip().decode('utf8') for line in fr.readlines()]
            for word_id, word in enumerate(lines):
                self.vocab_map[word] = word_id



if __name__ == '__main__':
    warplda = Warplda('./train.model.iter9900', '.train.vocab')

