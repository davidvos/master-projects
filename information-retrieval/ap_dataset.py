import os

import numpy as np
import pickle as pkl
import time
import random
from tqdm import tqdm

from collections import defaultdict, Counter


import download_ap
import read_ap


class APDataset():

    def __init__(self, window_size, vocab_size):

        # ensure dataset is downloaded
        download_ap.download_dataset()
        # pre-process the text
        docs_by_id = read_ap.get_processed_docs()

        self.word2id = dict()
        self.id2word = dict()

        self.window_size = window_size
        self.vocab_size = vocab_size

        self.docs_by_id = docs_by_id
        self.read_words(vocab_size)

    def read_words(self, vocab_size):
        word_frequency = Counter()

        for doc in self.docs_by_id:
            doc_words = self.docs_by_id[doc]
            for word in doc_words:
                word_frequency[word] += 1
        print('Read all words')

        self.word_frequency = word_frequency

        wid = 0
        vocab_words = word_frequency.most_common(vocab_size)
        for w, c in vocab_words:
            self.word2id[w] = wid
            self.id2word[wid] = w
            wid += 1
        print("Total embeddings: {}".format(self.vocab_size))

    def get_batch(self, batch_size):
        generator = self.get_generator()
        while True:
            batch = []
            for i in range(batch_size):
                batch.append(next(generator))
            yield batch

    def get_generator(self):
        doc_ids = reversed(list(self.docs_by_id.keys()))
        for doc_id in doc_ids:
            doc = self.docs_by_id[doc_id]
            n_words = len(doc)
            for index, word in enumerate(doc):
                if word in self.word2id.keys():
                    focus = self.word2id[word]
                    for window in range(1, self.window_size + 1):
                        try:
                            if index+window >= 0 and index+window <= n_words:
                                neg_context = [
                                    random.randint(0, self.vocab_size-1) for i in range(5)]
                                pos_context = self.word2id[doc[index-window]]
                                yield [focus, pos_context, neg_context]

                        except:
                            pass
                        try:
                            if index+window >= 0 and index+window <= n_words:
                                neg_context = [
                                    random.randint(0, self.vocab_size-1) for i in range(5)]
                                pos_context = self.word2id[doc[index+window]]
                                yield [focus, pos_context, neg_context]
                        except:
                            pass

    # def create_vocabulary(self, docs, vocab_file_name='vocabulary'):
    #     vocab_path = "./vocabulary.pkl"
    #     if os.path.exists(vocab_path):

    #         with open(vocab_path, "rb") as reader:
    #             vocab = pkl.load(reader)

    #         vocab_len = vocab["vocab_len"]
    #         vocab_words = vocab["vocab_words"]
    #         return vocab_len, vocab_words
    #     else:
    #         word_counter = Counter()

    #         doc_ids = list(docs.keys())

    #         print("Building Vocabulary")
    #         for doc_id in tqdm(doc_ids):
    #             doc = docs[doc_id]
    #             word_counter.update(doc)

    #         thresholded_vocab = Counter(
    #             {x: word_counter[x] for x in word_counter if word_counter[x] >= 50})

    #         vocab_words = [word for word, _ in thresholded_vocab.most_common()]
    #         vocab_len = len(vocab_words)

    #         with open(vocab_path, "wb") as writer:
    #             vocab = {
    #                 "vocab_len": vocab_len,
    #                 "vocab_words": vocab_words
    #             }
    #             pkl.dump(vocab, writer)
    #         return vocab_len, vocab_words


if __name__ == "__main__":
    dataset = APDataset(10, 50)
