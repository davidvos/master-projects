import argparse
import logging
from tabulate import tabulate

from eval import run

if __name__ == '__main__':
    FORMAT = '%(levelname)s: %(asctime)-15s - %(message)s'
    logging.basicConfig(filename='eval.log', level=logging.INFO, format=FORMAT)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-R", "--reference", help="reference translation", default='data/en/references/reference')
    argParser.add_argument("-H", "--hypothesis", help="hypothesis translation", default='data/en/hypothesis')
    argParser.add_argument("-lng", "--language", help="evaluated language", default='en')
    argParser.add_argument("-nr", "--num_refs", help="number of references", type=int, default=4)
    argParser.add_argument("-m", "--metrics", help="evaluation metrics to be computed", default='bleu,meteor,ter,chrf++,bert,bleurt')
    argParser.add_argument("-nc", "--ncorder", help="chrF metric: character n-gram order (default=6)", type=int, default=6)
    argParser.add_argument("-nw", "--nworder", help="chrF metric: word n-gram order (default=2)", type=int, default=2)
    argParser.add_argument("-b", "--beta", help="chrF metric: beta parameter (default=2)", type=float, default=2.0)

    args = argParser.parse_args()

    logging.info('READING INPUTS...')
    refs_path = args.reference
    hyps_path = args.hypothesis
    lng = args.language
    num_refs = args.num_refs
    metrics = args.metrics#.lower().split(',')

    nworder = args.nworder
    ncorder = args.ncorder
    beta = args.beta
    logging.info('FINISHING TO READ INPUTS...')

    result = run(refs_path=refs_path, hyps_path=hyps_path, num_refs=num_refs, lng=lng, metrics=metrics, ncorder=ncorder, nworder=nworder, beta=beta)
    
    metrics = metrics.lower().split(',')
    headers, values = [], []
    if 'bleu' in metrics:
        headers.append('BLEU')
        values.append(result['bleu'])

        headers.append('BLEU NLTK')
        values.append(round(result['bleu_nltk'], 2))
    if 'meteor' in metrics:
        headers.append('METEOR')
        values.append(round(result['meteor'], 2))
    if 'chrf++' in metrics:
        headers.append('chrF++')
        values.append(round(result['chrf++'], 2))
    if 'ter' in metrics:
        headers.append('TER')
        values.append(round(result['ter'], 2))
    if 'bert' in metrics:
        headers.append('BERT-SCORE P')
        values.append(round(result['bert_precision'], 2))
        headers.append('BERT-SCORE R')
        values.append(round(result['bert_recall'], 2))
        headers.append('BERT-SCORE F1')
        values.append(round(result['bert_f1'], 2))
    if 'bleurt' in metrics and lng == 'en':
        headers.append('BLEURT')
        values.append(round(result['bleurt'], 2))

    logging.info('PRINTING RESULTS...')
    print(tabulate([values], headers=headers))