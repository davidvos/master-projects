from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import csv
import pickle as pkl


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super(SkipGram, self).__init__()

        self.u_embeddings = nn.Embedding(
            vocab_size, embedding_size, sparse=True)
        self.v_embeddings = nn.Embedding(
            vocab_size, embedding_size, sparse=True)

        initrange = 1.0 / embedding_size
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, focus, pos_context, neg_context, batch_size):

        embed_u = self.u_embeddings(focus)
        embed_v = self.v_embeddings(pos_context)

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()

        neg_embed_v = self.v_embeddings(neg_context)

        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target + sum_log_sampled
        return -1*loss.sum()/batch_size

    def save_embedding(self, ix_to_word, file_name='saved_embeddings/skipgram_imp'):
        with open(file_name, 'wb') as embed_file:
            embedding = {}
            embedding_matrix = self.u_embeddings.weight.data
            for word_index in range(len(embedding_matrix)):
                word = ix_to_word[word_index]
                embedding[word] = list(embedding_matrix[word_index].numpy())
            pkl.dump(embedding, embed_file)
