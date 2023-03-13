
import os
import json
import pickle as pkl
from collections import defaultdict, Counter

from ast import literal_eval

import numpy as np
from tqdm import tqdm

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel

import numpy as np

class GensimCorpus():
    """
    This class implements a corpus which can be saved and transformed for efficiency.
    """

    def __init__(self, docs_by_id, corpus_index_path="./saved_models/gensim-corpus.pkl", embedding="tfidf"):

        tfidf_model_path = "./saved_models/tfidf-gensim-model.mm"
        dictionary_path = "./saved_models/gensim_dictionary.mm"

        self.tfidf_corpus_path = "./saved_models/tfidf-corpus.crp"
        self.corpus_path = "./saved_models/corpus.crp"

        doc_index_path = "./saved_models/doc_ids.pkl"

        # Check if dictionary file exists, otherwise create it.
        if os.path.exists(dictionary_path):
            with open(dictionary_path, "rb") as reader:
                data = pkl.load(reader)
                self.dictionary = data["dictionary"]
        else:
            # Create a dictionary representation of the documents.
            self.dictionary = Dictionary(docs_by_id.values())
            self.dictionary.filter_extremes(no_below=150, no_above=0.5)

            with open(dictionary_path, "wb") as writer:
                pkl.dump({
                    "dictionary": self.dictionary
                    }, writer)

        self.bow_corpus = BOWCorpus(docs_by_id, self.dictionary)

        if os.path.exists(tfidf_model_path):
            with open(tfidf_model_path, "rb") as reader:
                data = pkl.load(reader)
                self.tfidf_model = data["model"]
        else:
            print("Building TFIDF model")
            self.tfidf_model = TfidfModel(self.bow_corpus)

            with open(tfidf_model_path, "wb") as writer:
                pkl.dump({
                    "model": self.tfidf_model
                    }, writer)

        self.tfidf_corpus = TFIDFCorpus(self.bow_corpus, self.tfidf_model)

        if os.path.exists(doc_index_path):
            print("Reading doc ids")
            with open(doc_index_path, "rb") as reader:
                data = pkl.load(reader)
                self.doc_ids = data["doc-ids"]

        else:
            self.doc_ids = list(docs_by_id.keys())
            print("Collecting doc ids")
            with open(doc_index_path, "wb") as writer:
                pkl.dump({
                    "doc-ids": self.doc_ids
                    }, writer)

        self.embedding = embedding

    def get_corpus(self):
        
        if self.embedding == "tfidf":
            return self.tfidf_corpus
        else:
            return self.bow_corpus


class BOWCorpus():

    def __init__(self, docs_by_id, dictionary, path="./saved_models/bow_corpus.crp"):

        self.path = path

        if not os.path.exists(self.path):

            print("creating bow repr.")
            corpus = [dictionary.doc2bow(doc) for doc in docs_by_id.values()]
            print("saving bow repr to disk.")
            MmCorpus.serialize(self.path, corpus)

        self.corpus = MmCorpus(self.path)

    def __iter__(self):
        """
        Iter over corpus.
        """
        for doc in tqdm(self.corpus):
            yield doc
    
    def __len__(self):
        return len(self.corpus)

class TFIDFCorpus():

    def __init__(self, bow_docs, tfidf_model, path="./saved_models/tfidf_corpus.crp"):

        self.path = path

        if not os.path.exists(self.path):

            print("creating tfidf repr.")
            corpus = [tfidf_model[doc] for doc in bow_docs]
            print("saving tfidf repr to disk.")
            MmCorpus.serialize(self.path, corpus)

        self.corpus = MmCorpus(self.path)

    def __iter__(self):
        for doc in tqdm(self.corpus):
            yield doc
    
    def __len__(self):
        return len(self.corpus)

class ModelCorpus():

    def __init__(self, docs, model, persist=False, path="./saved_models/lsi_corpus.crp"):

        self.path = path
        self.persist = persist
        self.model = model

        if not os.path.exists(self.path) and self.persist:

            print("creating model repr.")
            corpus = model[docs]
            print("saving model repr to disk.")
            MmCorpus.serialize(self.path, corpus)

        if not self.persist:
            self.corpus = docs
        else:
            self.corpus = MmCorpus(self.path)

    def __iter__(self):
        for doc in tqdm(self.corpus):
            if not self.persist:
                yield self.model[doc]
            else:
                yield doc

    def __len__(self):
        return len(self.corpus)