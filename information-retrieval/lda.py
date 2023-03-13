import os
import json
import pickle as pkl
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

import read_ap
import download_ap

from gensim_corpus import GensimCorpus, BOWCorpus, TFIDFCorpus, ModelCorpus
from gensim.models import LdaModel
from gensim.matutils import kullback_leibler, sparse2full

from trec import TrecAPI

import argparse

class LatentDirichletAllocation():
    """
    This class implements latent dirichlet allocation using the gensim library.
    """
    def __init__(self, corpus, num_topics=500, iterations=2000, passes=20, eval_every=None, embedding="bow"):

        self.lda_model_path = "./saved_models/gensim-lda-model-nt-{}.mm".format(num_topics)
        self.lda_corpus_path = "./saved_models/gensim-lda-nt-{}-corpus.crp".format(num_topics)

        self.index_path = "./saved_models/gensim-lda-model.pkl"
        self.lda_corpus_path = "./saved_models/gensim-lda-corpus.crp"

        self.corpus = corpus
        self.num_topics = num_topics
        self.lda_corpus = []

        if os.path.exists(self.lda_model_path):
            print("LDA model already trained, loading from disk.")
            self.model = LdaModel.load(self.lda_model_path)

        else:

            # Make a index to word dictionary.
            temp = corpus.dictionary[0]  # This is only to "load" the dictionary.
            id2word = corpus.dictionary.id2token

            print("Training LDA model.")
            self.model = LdaModel(
                corpus=list(corpus.get_corpus()),
                id2word=id2word,
                iterations=iterations,
                num_topics=num_topics,
                eval_every=eval_every
            )

            self.model.save(self.lda_model_path)

        self.lda_corpus = ModelCorpus(corpus.get_corpus(), self.model, path=self.lda_corpus_path, persist=True)

        self.lda_corpus_pers = [sparse2full(doc, self.num_topics) for doc in self.lda_corpus]

    def search(self, query):

        query_repr = read_ap.process_text(query)
        vec_query = self.corpus.dictionary.doc2bow(query_repr)
        lda_query = sparse2full(self.model[vec_query], self.num_topics)

        results = defaultdict(float)
        for doc_id, lda_doc_repr in zip(self.corpus.doc_ids, self.lda_corpus_pers):
            results[doc_id] = kullback_leibler(lda_query, lda_doc_repr)

        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
        return list(results.items())

if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models/sim_temps", exist_ok=True)
    os.makedirs("raw_output", exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("-embedding", type=str, default="tfidf", help="Embedding to use in training LDA.")
    parser.add_argument("-num_topics", type=int, default=500, help="Number of topics to use in training LDA.")
    args = parser.parse_args()

    num_topics = args.num_topics
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = None
    docs_by_id = read_ap.get_processed_docs()

    gensim_corpus = GensimCorpus(docs_by_id, args.embedding)

    lda = LatentDirichletAllocation(gensim_corpus, eval_every=None, num_topics=num_topics, embedding=args.embedding)

    # read in the qrels
    qrels, queries = read_ap.read_qrels()

    overall_ser = {}

    print("Running LDA benchmark")

    # Write results to trec-style file
    if not os.path.exists("lda_results.out"):

        # collect results
        for qid in tqdm(qrels):
            query_text = queries[qid]

            results = lda.search(query_text)
            overall_ser[qid] = dict(results)

        results_lines = []
        for qid in overall_ser:
            for doc_id in overall_ser[qid]:
                results_lines.append(str(qid) + '\tQO\t' + doc_id + '\t0\t' + str(overall_ser[qid][doc_id]) + '\tSTANDARD\n')
        with open('./raw_output/lda_results.out', 'w') as f:
            f.writelines(results_lines)

    trec = TrecAPI('D:/Google Drive/Documenten/UVA/MSc AI/Information Retrieval 1/trec_eval-master/trec_eval.exe')
    metrics = trec.evaluate(test_file_name='datasets/ap/qrels.tsv', prediction_file_name='./raw_output/lda_results.out', metrics_to_capture={'map', 'ndcg'})

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open("./results/lda-{}-topics.json".format(num_topics), "w") as writer:
        json.dump(metrics, writer, indent=1)