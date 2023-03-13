import os
import json
import pickle as pkl
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from pathlib import Path

import read_ap
import download_ap

# from gensim_corpus import GensimCorpus, CorpusFilePipe
from gensim_corpus import GensimCorpus, BOWCorpus, TFIDFCorpus, ModelCorpus
from gensim.models import LsiModel
from gensim import similarities
from gensim.test.utils import get_tmpfile

from gensim.matutils import kullback_leibler, sparse2full

from trec import TrecAPI

import argparse

class LatentSemanticIndexing():
    """
    This class implements Latent semantic indexing using the genims library.
    """
    def __init__(self, corpus, embedding="bow", num_topics=500, chunksize=20000):

        self.lsi_model_path = "./saved_models/gensim-lsi-{}-model-nt-{}.mm".format(embedding, num_topics)
        self.lsi_corpus_path = "./saved_models/gensim-{}-lsi-nt-{}-corpus.crp".format(embedding, num_topics)
        self.sim_matrix_path = "./saved_models/sim-matrix-{}-{}.mm".format(embedding, num_topics)
        self.sim_matrix_temp_path = "./saved_models/sim_temps/sim_temp-{}-{}.tmp".format(embedding, num_topics)

        self.embedding = embedding
        self.corpus = corpus
        self.num_topics = num_topics

        if os.path.exists(self.lsi_model_path):

            print("LSI {} model already trained, loading from disk.".format(embedding))
            self.model = LsiModel.load(self.lsi_model_path)

        else:

            # Make a index to word dictionary.
            temp = corpus.dictionary[0]  # This is only to "load" the dictionary.
            id2word = corpus.dictionary.id2token

            print("Training LSI model.")
            self.model = LsiModel(
                corpus=list(corpus.get_corpus()),
                id2word=id2word,
                chunksize=chunksize,
                num_topics=num_topics
            )
            print("Saving LSI model.")
            self.model.save(self.lsi_model_path)

        self.lsi_corpus = ModelCorpus(corpus.get_corpus(), self.model, path=self.lsi_corpus_path)

        if os.path.exists(self.sim_matrix_path):
            print("Similarities matrix {} model already trained, loading from disk.".format(embedding))
            self.index = similarities.Similarity.load(self.sim_matrix_path)
        else:
            print("Creating similarities index.")
            Path(self.sim_matrix_temp_path).touch(exist_ok=True)
            self.index = similarities.Similarity(self.sim_matrix_temp_path, self.lsi_corpus, num_features=self.num_topics)
            self.index.save(self.sim_matrix_path)

    def search(self, query):

        query_repr = read_ap.process_text(query)
        vec_query = self.corpus.dictionary.doc2bow(query_repr)

        if self.embedding == "bow":
            lsi_query = self.model[vec_query]
        elif self.embedding == "tfidf":
            lsi_query = self.model[self.corpus.tfidf_model[vec_query]]

        sims = self.index[lsi_query]
        sims = sorted(zip(self.corpus.doc_ids, sims), key=lambda item: -item[1])
        return sims

if __name__ == "__main__":

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./saved_models/sim_temps", exist_ok=True)
    os.makedirs("./raw_output", exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("-embedding", type=str, default="tfidf", help="Embedding to use in training LDA.")
    parser.add_argument("-num_topics", type=int, default=500, help="Number of topics to use in training LDA.")
    parser.add_argument("--evaluate", default=False, action="store_true")
    args = parser.parse_args()

    evaluate = args.evaluate
    embedding = args.embedding
    num_topics = args.num_topics

    embeddings = ["tfidf", "bow"]
    num_topics_search = [10, 50, 100]
    search_n_topics = False

    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = None
    docs_by_id = read_ap.get_processed_docs()

    if search_n_topics:
        for embedding in embeddings:
            gensim_corpus = GensimCorpus(docs_by_id, embedding=embedding)

            for num_topics in num_topics_search:
                print("BUILDING FOR {} TOPICS {}".format(num_topics, embedding))
                lda = LatentSemanticIndexing(gensim_corpus, embedding=embedding, num_topics=num_topics)
                if evaluate:
                    # read in the qrels
                    qrels, queries = read_ap.read_qrels()

                    overall_ser = {} 

                    print("Running LSI benchmark")

                    # collect results
                    for qid in tqdm(qrels): 
                        query_text = queries[qid]

                        results = lda.search(query_text)
                        overall_ser[qid] = dict(results)

                    if not os.path.exists("lsi-{}-{}-topics_results.out".format(embedding, num_topics)):
                        # Write results to trec-style file
                        results_lines = []
                        for qid in overall_ser:
                            for doc_id in overall_ser[qid]:
                                results_lines.append(str(qid) + '\tQO\t' + doc_id + '\t0\t' + str(overall_ser[qid][doc_id]) + '\tSTANDARD\n')
                        with open('./raw_output/lsi-{}-{}-topics_results.out'.format(embedding, num_topics), 'w') as f:
                            f.writelines(results_lines)

                    trec = TrecAPI('D:/Google Drive/Documenten/UVA/MSc AI/Information Retrieval 1/trec_eval-master/trec_eval.exe')
                    metrics = trec.evaluate(test_file_name='datasets/ap/qrels.tsv', prediction_file_name='./raw_output/lsi-{}-{}-topics_results.out'.format(embedding, num_topics), metrics_to_capture={'map', 'ndcg'})

                    # dump this to JSON
                    # *Not* Optional - This is submitted in the assignment!
                    with open("./results/lsi-{}-embedding-{}-topics.json".format(embedding, num_topics), "w") as writer:
                        json.dump(metrics, writer, indent=1)

    else:
        gensim_corpus = GensimCorpus(docs_by_id, embedding=embedding)
        lda = LatentSemanticIndexing(gensim_corpus, embedding=embedding, num_topics=num_topics)
