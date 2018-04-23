#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 hschen0712 <hschen0712@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Collapsed Gibbs Sampling
"""
import numpy as np
import json
from model import Warplda
from collections import Counter

def gibbs_sampling(doc, model, num_iter=10):
    '''
    gibbs采样
    :param doc: 预处理后的文档
    :param model: warplda模型
    :param num_iter: 迭代次数，默认为10
    :return: 
    '''
    vocab_map = model.vocab_map
    doc = [vocab_map[word] for word in doc.split() if word in vocab_map]

    K = model.num_topics
    num_tokens = len(doc)
    alpha = model.alpha
    alpha_bar = model.alpha_bar
    beta = model.beta
    beta_bar = model.beta_bar
    Cvk = model.Cvk
    Ck = model.Ck
    z = np.zeros((num_tokens, K))
    # 随机初始化每个token的指派
    for n in range(num_tokens):
        rand_topic = np.random.randint(0, K)
        z[n, rand_topic] = 1

    for i in range(num_iter):
        for n, word_id in enumerate(doc):
            pz = np.divide(np.multiply(z.sum(axis=0) + alpha, Cvk[word_id, :] + beta), Ck + beta_bar)
            k = np.random.multinomial(1, pz / pz.sum()).argmax()
            z[n, :] *= 0
            z[n, k] = 1
    topic_cnt = Counter(z.argmax(axis=1))
    topic_dist = [(topic_id, (cnt + alpha)/(num_tokens + alpha_bar)) for topic_id, cnt in topic_cnt.iteritems()]
    topic_dist = json.dumps(dict([(topic_id, prob) for topic_id, prob in topic_dist if prob >=0.05 ]))

    return topic_dist


if __name__ == '__main__':
    from time import time

    docs = [line.strip().decode('utf8') for line in open('./test_corpus_for_lda.dat').readlines()]

    warplda = Warplda('./train.model.iter9900', './train.vocab')
    start = time()
    for d, doc in enumerate(docs):
        print 'doc {}'.format(d)
        topic_dist = gibbs_sampling(doc, warplda)
        print topic_dist
    stop = time()

    print stop - start
