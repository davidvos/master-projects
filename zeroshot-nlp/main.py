from transformers import BertForMaskedLM, BertTokenizer
import torch
import polyglot
from polyglot.text import Text, Word
from word2word import Word2word
from embedding_transform import EmbeddingTransform

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
import json
# import nltk
import spacy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CrossLingualPipeline:
    
    def __init__(self, transform_type="proj", save_fn="./models/stored_embeddings-{}.pkl", transform_subset=None):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(DEVICE)
        self.dutch_bert_tokenizer = BertTokenizer.from_pretrained('wietsedv/bert-base-dutch-cased')
        self.dutch_bert_model = BertForMaskedLM.from_pretrained('wietsedv/bert-base-dutch-cased').to(DEVICE)

        self.transform_type = transform_type

        self.nl2en = Word2word("nl", "en")

        # get token embeddings for start, end and mask token, to use for masking when overriding embeddings
        tokens = "[MASK]"
        emb_tokens = self.bert_model.bert.embeddings.word_embeddings(torch.tensor([self.bert_tokenizer.encode(tokens, add_special_tokens=True)], device=DEVICE))
        self.srt_tok = emb_tokens[0, 0, :]
        self.msk_tok = emb_tokens[0, 1, :]
        self.end_tok = emb_tokens[0, 2, :]
        
        if transform_subset:
            save_fn = save_fn.format(transform_subset)
        else:
            save_fn = save_fn.format("all")

        if not os.path.exists(save_fn):

            with open('data/filtered_en_nl_dict.txt', 'r', encoding="utf-8") as f:
                lines = f.read().splitlines()
                en_words_to_embed = [words.split('\t')[1] for words in lines]
                nl_words_to_embed = [words.split('\t')[0] for words in lines]

            if transform_subset:
                # Only use a fraction of the most common words when calculating the
                # transformation
                with open('data/train.txt', 'r', encoding="utf-8") as f:
                    lines = f.read().splitlines()

                tokens = []
                for line in lines:
                    tokens += [token.lower() for token in line.split()]
                token_counts = Counter(tokens)
                filt_token_counts = Counter({k: token_counts.get(k, 0) for k in nl_words_to_embed}).most_common(int(len(nl_words_to_embed) * transform_subset))
                most_common_subset = [w[0] for w in filt_token_counts]

                most_common_ind = []
                for word in most_common_subset:
                    most_common_ind.append(nl_words_to_embed.index(word))

                new_nl_words_to_embed = [nl_words_to_embed[i] for i in most_common_ind]
                new_en_words_to_embed = [en_words_to_embed[i] for i in most_common_ind]

                nl_words_to_embed = new_nl_words_to_embed
                en_words_to_embed = new_en_words_to_embed

            english_embeddings = self.get_english_embeddings(en_words_to_embed)
            dutch_embeddings = self.get_dutch_embeddings(nl_words_to_embed)

            with open(save_fn, "wb") as f:
                pickle.dump({"eng_emb":english_embeddings, "dut_emb":dutch_embeddings}, f)
        else:
            with open(save_fn, "rb") as f:
                fc = pickle.load(f)
                english_embeddings = fc["eng_emb"]
                dutch_embeddings = fc["dut_emb"]

        self.transform = EmbeddingTransform(self.transform_type, dutch_embeddings, english_embeddings, str_tok=self.srt_tok, end_tok=self.end_tok)

    def get_english_embeddings(self, words_to_embed):
        """
        Read the bilingual dictionary and embed all individual words in english.
        """
        word_indices = torch.tensor(self.bert_tokenizer.encode(words_to_embed, add_special_tokens=True), device=DEVICE).unsqueeze(0)
        all_word_embeddings = self.bert_model.bert.embeddings.word_embeddings
        english_embeddings = all_word_embeddings(word_indices)
        # filter out start and end tokens
        return english_embeddings.squeeze()[1:-1]

    def get_dutch_embeddings(self, words_to_embed):
        """
        Read the bilingual dictionary and embed all individual words in dutch.
        """
        dutch_embeddings = []
        for word in words_to_embed:
            try:
                poly_word = Word(word, language="nl")
                dutch_embeddings.append(poly_word.vector)
            except KeyError:
                continue
        print(f"number of dutch embeddings: {len(dutch_embeddings)}")
        return torch.tensor(dutch_embeddings, device=DEVICE, dtype=torch.float)

    def get_dutch_sentence_embedding(self, sentence):
        """
        Return word-for-word embedding of dutch sentence
        """

        # remove these strange edge cases
        sentence = sentence.replace("''", "")
        sentence = sentence.replace('``', "")
        # double whitespaces confuse the model, remove these.
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")

        sent_tokens = sentence.split()
        emb_sent = []
        used_tokens = []
        for word in sent_tokens:
            try:
                poly_word = Word(word.lower(), language="nl")
                emb_sent.append(poly_word.vector)
                used_tokens.append(word)
            except KeyError:
                continue

        sent_tokens.sort(key=len, reverse=True)
        for word in sent_tokens:
            if word not in used_tokens:
                sentence = sentence.replace(word, '')

        # double whitespaces confuse the model, remove these.
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")

        return torch.tensor(emb_sent, device=DEVICE, dtype=torch.float), sentence

    def map_dutch_to_english(self, sentence):
        """
        Transform a dutch sentence to an embedded english sentence
        """
        emb_dutch, sentence = self.get_dutch_sentence_embedding(sentence)
        return self.transform.dutch_to_english(emb_dutch), sentence

    def get_sentence_perplexity(self, sentence, tokenizer, model, override_embeddings=None):
        """
        Calculate the perplexity of a given sentence, using given model and tokenizer

        If override_embeddings is specified, these embeddings will be used instead of the embeddings
        created by BERT.
        """
        model.eval()
        with torch.no_grad():
            if override_embeddings is None:
                word_indices = tokenizer.encode(sentence)
            else:
                sentence_list = []
                for word in sentence.split():
                    try:
                        sentence_list.append(self.nl2en(word.lower())[0])
                    except KeyError as ke:
                        sentence_list.append(word.lower())
                word_indices = tokenizer.encode(sentence_list)

            mask_token = tokenizer.encode("[MASK]")[1]
            mask_embedding = self.msk_tok

            all_word_indices = torch.zeros(len(word_indices)-2, len(word_indices), device=DEVICE).long()
            all_word_labels = torch.zeros(len(word_indices)-2, len(word_indices), device=DEVICE).fill_(-100).long()
            all_segment_tensors = torch.zeros(len(word_indices)-2, len(word_indices), device=DEVICE).long()
            if override_embeddings is not None:
                all_embeddings = torch.zeros(len(word_indices)-2, *override_embeddings.shape, device=DEVICE).float()

            word_indices = torch.tensor(word_indices, device=DEVICE)

            for mask_index in range(1, len(word_indices)-1):
                all_word_indices[mask_index - 1] = word_indices
                all_word_indices[mask_index - 1, mask_index] = mask_token
                all_word_labels[mask_index - 1, mask_index] = word_indices[mask_index]
                if override_embeddings is not None:
                    all_embeddings[mask_index - 1] = override_embeddings
                    all_embeddings[mask_index - 1, mask_index] = mask_embedding

            if override_embeddings is not None:
                prediction_scores = model(token_type_ids=all_segment_tensors, inputs_embeds=all_embeddings)
            else:
                prediction_scores = model(all_word_indices, token_type_ids=all_segment_tensors)

            prediction_scores = prediction_scores[0]

            del all_word_indices
            del all_segment_tensors

            ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
            losses = ce_loss(prediction_scores.view(-1, model.config.vocab_size), all_word_labels.view(-1))

            del all_word_labels

            losses = losses[losses.nonzero()]
            conf = []
            predicted_token_idxs = []

            vals, args = torch.max(prediction_scores, dim=2)

            for mask_index in range(1, len(word_indices)-1):

                predicted_token_idxs.append(args[mask_index-1, mask_index].item())
                conf.append(round(vals[mask_index-1, mask_index].item(), 2))

            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_idxs)
            actual_tokens = tokenizer.convert_ids_to_tokens(word_indices)

            #fprint('Predicted: {}\n, Actual: {}\n\n'.format(predicted_tokens, actual_tokens))
            perplexity = torch.exp(losses).view(-1).cpu().numpy().tolist()
            return losses.cpu().numpy().tolist(), conf, predicted_tokens, actual_tokens, perplexity


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--transform_type", choices=["SGD", "SVD", "CCA"], type=str, default="SVD")
    argparser.add_argument("--subset_for_transform", type=float, help="Specifies a fraction of the most common words to use to estimate the transformation.")
    argparser.add_argument("--cuda", help="Run on cuda", action="store_true", default=False)
    argparser.add_argument("--sentence_length_cutoff", help="sentence longer than this value get discarded", type=int, default=50)
    argparser.add_argument("--path_to_dataset", help="specify an alternative dataset to evaluate", type=str, default='data/valid.txt')
    argparser.add_argument("--native", help="evaluate the validation dataset on the native model", action="store_true", default=False)

    args = argparser.parse_args()

    cross_lin_pipe = CrossLingualPipeline(transform_type=args.transform_type, transform_subset=args.subset_for_transform)
    # cross_lin_pipe.compare_embeddings()

    with open(args.path_to_dataset, 'r', encoding="utf-8") as f:
        sentences = f.read().splitlines()

    model_name = "cross_lingual" if not args.native else "native"
    dataset_name = args.path_to_dataset.rstrip(".txt").rsplit("/", 1)[1]
    f = open(f"./experiments/{model_name}_{args.transform_type}_{args.subset_for_transform}_cutoff_{dataset_name}.txt", "a+")

    sent_index = 0

    for sentence in tqdm(sentences):

        # discard sentences longer than a given number of words, reduce computational complexity
        if len(sentence.split()) > args.sentence_length_cutoff:
            sent_index += 1
            continue

        # sentence = cross_lin_pipe.shuffle_pos(sentence)
        original_sentence = str(sentence)
        transformed_embedding, sentence = cross_lin_pipe.map_dutch_to_english(sentence)
        # transfered perplexity
        if not args.native:
            losses, conf, predicted_tokens, actual_tokens, perplexity = cross_lin_pipe.get_sentence_perplexity(sentence, cross_lin_pipe.bert_tokenizer, cross_lin_pipe.bert_model, override_embeddings=transformed_embedding)
        else:
            # native perplexity
            losses, conf, predicted_tokens, actual_tokens, perplexity = cross_lin_pipe.get_sentence_perplexity(original_sentence, cross_lin_pipe.dutch_bert_tokenizer, cross_lin_pipe.dutch_bert_model)

        sent_result = {
            "sentence":sentence,
            "sentence_index": sent_index,
            "model_type": model_name,
            "ce_loss_per_token": losses,
            "conf_per_token": conf,
            "predicted_tokens": predicted_tokens,
            "actual_tokens": actual_tokens,
            "perplexity": perplexity
        }

        sent_index += 1

        f.write(json.dumps(sent_result) + '\n')
    f.close()

