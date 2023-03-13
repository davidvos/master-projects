from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os
import pytrec_eval
from tqdm import tqdm
import json

from SkipGram import SkipGram
from ap_dataset import APDataset
from read_ap import process_text, read_qrels

import pickle as pkl


class Word2VecRetrieval():

    def __init__(self, window_size, vocab_size, embedding_size, batch_size, model_path):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        if os.path.isfile("word2vec_dataset.p"):
            self.dataset = pkl.load(open("word2vec_dataset.p", "rb"))
        else:
            self.dataset = APDataset(window_size, vocab_size)
            pkl.dump(self.dataset, open("word2vec_dataset.p", "wb"))

        self.data_generator = self.dataset.get_batch(self.batch_size)

        self.skip_gram = SkipGram(self.dataset.vocab_size, embedding_size)

        if os.path.isfile(model_path):
            self.skip_gram.load_state_dict(torch.load(model_path))
            self.skip_gram.eval()

    def train(self):

        optimizer = torch.optim.SparseAdam(self.skip_gram.parameters())

        self.skip_gram.train()

        print('Training...')

        step = 0

        for batch in self.data_generator:

            step += 1
            focus = [sample[0] for sample in batch]
            focus = torch.tensor(focus)

            pos_context = [sample[1] for sample in batch]
            pos_context = torch.tensor(pos_context)

            neg_context = [sample[2] for sample in batch]
            neg_context = torch.tensor(neg_context)

            optimizer.zero_grad()
            loss = self.skip_gram.forward(
                focus, pos_context, neg_context, self.batch_size)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Step: ', step, ', Loss: ', loss)

            if step % 20000 == 0:
                torch.save(self.skip_gram.state_dict(),
                           './saved_models/word2vec.pt'.format(step))
                self.skip_gram.save_embedding(
                    self.dataset.id2word, './word2vec_embedding.pkl')

    def find_similar_words(self, query, n=11):
        word_to_vec = pkl.load(open("word2vec_embedding.pkl", "rb"))
        query = process_text(query)[0]
        word_vec = word_to_vec[query]
        distances = []
        for index, word_key in enumerate(word_to_vec):
            cos_sim = np.dot(word_vec, word_to_vec[word_key]) / (
                np.linalg.norm(word_vec) * np.linalg.norm(word_to_vec[word_key]))
            distances.append((index, word_key, cos_sim))
        sorted_by_distance = sorted(
            distances, reverse=True, key=lambda tup: tup[2])
        for matching_word in sorted_by_distance[:n]:
            print(matching_word[1])

    def embed_query(self, word_to_vec, query, aggregation='mean'):

        query_repr = process_text(query)

        doc = []
        for query_term in query_repr:
            if query_term not in word_to_vec:
                continue
            else:
                doc.append(word_to_vec[query_term])

        if aggregation == 'mean':
            doc = np.mean(doc, axis=0)
        return doc

    def embed_doc(self, doc, word_to_vec, aggregation='mean'):

        doc_repr = []
        for doc_term in doc:
            if doc_term not in word_to_vec:
                continue
            else:
                doc_repr.append(word_to_vec[doc_term])

        if aggregation == 'mean':
            doc_repr = np.mean(doc_repr, axis=0)
        return doc_repr

    def make_doc_repr(self):
        word_to_vec = pkl.load(open("word2vec_embedding.pkl", "rb"))
        doc_reprs = []
        for doc_id in self.dataset.docs_by_id:
            print(doc_id)
            doc = self.dataset.docs_by_id[doc_id]
            doc_embed = self.embed_doc(doc, word_to_vec)
            doc_reprs.append((doc_id, doc_embed))
        pkl.dump(doc_reprs, open("doc_embeds.p", "wb"))

    def search(self, doc_embeds, query):

        word_to_vec = pkl.load(open("word2vec_embedding.pkl", "rb"))
        query_ranking = []
        query_embed = self.embed_query(word_to_vec, query)

        for doc_id, doc_embed in doc_embeds:
            similarity = np.dot(query_embed, doc_embed) / \
                (np.linalg.norm(query_embed)*np.linalg.norm(doc_embed))
            query_ranking.append((doc_id, float(similarity)))
        sorted_by_distance = sorted(
            query_ranking, reverse=True, key=lambda tup: tup[1])
        return sorted_by_distance


if __name__ == '__main__':
    skipgram_search = Word2VecRetrieval(
        5, 25000, 200, 1000, 'word2vec.pkl')

    # skipgram_search.train()
    # skipgram_search.skipgram_search('art', 11)
    doc_representations = skipgram_search.make_doc_repr()

    qrels, queries = read_qrels()
    doc_embeds = pkl.load(open("word2vec.p", "rb"))

    overall_ser = {}

    print("Running Skipgram Benchmark")
    # collect results
    for qid in list(tqdm(qrels)):
        print('Benchmark: ' + str(qid))
        query_text = queries[qid]
        results = skipgram_search.search(doc_embeds, query_text)
        overall_ser[qid] = dict(results)
    # run evaluation with `qrels` as the ground truth relevance judgements
    # here, we are measuring MAP and NDCG, but this can be changed to
    # whatever you prefer
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    results_lines = []
    for qid in overall_ser:
        for doc_id in overall_ser[qid]:
            results_lines.append(str(qid) + '\tQO\t' + doc_id +
                                 '\t0\t' + str(overall_ser[qid][doc_id]) + '\tSTANDARD\n')
        print('Writing: ' + str(qid))
    with open('skipgram_results.out', 'w') as f:
        f.writelines(results_lines)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open("word2vec.json", "w") as writer:
        json.dump(metrics, writer, indent=1)
