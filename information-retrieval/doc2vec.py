import os
import gensim
import read_ap
import argparse
from tqdm import tqdm
from trec import TrecAPI
import json
import smart_open


def train(config):
    print(f"Training vec dim: {config.vector_dim},  window size: {config.window_size}, Vocab size: {config.vocab_size}")
    if not os.path.exists(config.model_file) or config.t:
        print("\n###    Reading in the documents    ###\n")
        docs_by_id = read_ap.get_processed_docs()

        print("\n### Converting to gensim standards ###\n")
        train_docs = list(AP2Gensim(docs_by_id))

        model = gensim.models.doc2vec.Doc2Vec(vector_size=config.vector_dim, window=config.window_size, min_count=config.min_count, dm=0,
                                                max_vocab_size=config.vocab_size, epochs=config.epochs)

        print("\n###         Building vocab         ###\n")
        model.build_vocab(train_docs)

        print("\n###         Training model         ###\n")
        model.train(train_docs, total_examples=model.corpus_count, epochs=model.epochs)
        
        print("\n###          Saving model          ###\n")
        model.save(config.model_file)
    else:
        print("A model already exists so skipping training")

def display_result(results):
    docs, ids = read_ap.read_ap_docs()
    res_tex = []
    for res in results:
        res_tex.append(docs[ids.index(res[0])])
    del docs, ids
    for i, res in enumerate(res_tex):
        print(f"\n##### Result : {i+1} #####\n")
        print(res)

def search(config):
   
    if not os.path.exists(config.model_file):
        raise ValueError("no model available for search, try setting '-t' to true to train model first")
    else:
        model = gensim.models.doc2vec.Doc2Vec.load(config.model_file)
        
    query = read_ap.process_text(config.search)
    vector = model.infer_vector(query)
    most_similar = model.docvecs.most_similar([vector], topn=config.top_n)

    display_result(most_similar)
    return most_similar

def AP2Gensim(docs_dict):
    for key, value in docs_dict.items():
        yield gensim.models.doc2vec.TaggedDocument(value, [key])



def evaluate(config, qrels, queries):

    if not os.path.exists(config.model_file):
        raise ValueError("no model available for search, try setting '-t' to true to train model first")
    else:
        model = gensim.models.doc2vec.Doc2Vec.load(config.model_file)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    # read in the qrels
    overall_ser = {}

    print("Running TFIDF Benchmark")
    # collect results
    results_lines = []
    for qid in tqdm(qrels): 
        query_text = queries[qid]

        vector = model.infer_vector(read_ap.process_text(query_text))
        results = model.docvecs.most_similar([vector], topn=164557)
        to_write = [str(qid)+ '\tQO\t' + doc_id + '\t0\t' + str(score) + '\tSTANDARD\n' for doc_id, score in results]
        
        with smart_open.open(config.write_file, 'a') as f:
            f.writelines(to_write)


def trec_eval(file_name):
    string = 'D:/trec_eval-master/trec_eval-master/trec_eval.exe'
    trec = TrecAPI(string)
    metrics = trec.evaluate(test_file_name='./datasets/ap/qrels.tsv', prediction_file_name=file_name, metrics_to_capture={'map', 'ndcg'})
    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open(file_name[:-4]+"_trec.out", "w") as writer:
        json.dump(metrics, writer, indent=1)


if __name__ == "__main__":

     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--window_size', type=int, default=5, help="Window size for computing embedding")
    parser.add_argument('--vocab_size', type=int, default=200000, help='Maximum vocabulary size')
    parser.add_argument('--vector_dim', type=int, default=300, help='Vector dimension of embedding')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train doc2vec')
    parser.add_argument('--min_count', type=int, default=2, help='Minimal times a word has to occur to be considered')

    parser.add_argument('--model_file', type=str, default="./doc2vec.p", help='Number of examples to process in a batch')
    parser.add_argument('--search', type=str, default=None, help='If True perform search of gives search query')
    parser.add_argument('-t', type=bool, default=True, help='If True trains doc2vec and saves to --model_file otherwise tries to load model from --model_file')
    parser.add_argument('--top_n', type=int, default=10, help='Amount of results for search query')

    config = parser.parse_args()

   
    if config.t:
      train(config)
    if config.search != None:
      search(config)

