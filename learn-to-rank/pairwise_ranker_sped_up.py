import torch
import sys
import os

import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import random

import itertools


class RankNet(torch.nn.Module):

    def __init__(self, input_dim=501, architecture=[64, 32], output_dim=1):
        super(RankNet, self).__init__()

        self.layers = torch.nn.ModuleList()

        prev_layer = input_dim
        for layer in architecture:
            self.layers.append(torch.nn.Linear(prev_layer, layer))
            self.layers.append(torch.nn.LeakyReLU(0.2))
            prev_layer = layer
        self.layers.append(torch.nn.Linear(prev_layer, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def evaluate(net, eval_data):

    feat_matrix = torch.FloatTensor(eval_data.feature_matrix).to('cpu')
    labels = torch.FloatTensor(eval_data.label_vector).to('cpu')

    n_queries = eval_data.num_queries()

    net.eval()

    scores = np.zeros(labels.shape)

    eval_size = 10
    batch_size = 1024
    with torch.no_grad():

        for iteration, batch_start in enumerate(list(range(0, len(feat_matrix), batch_size))):
            batch_inputs = feat_matrix[batch_start: batch_start + batch_size]

            out = net(batch_inputs)
            scores[batch_start: batch_start +
                   batch_size] = out.detach().squeeze().cpu().numpy()

        for qid in range(0, min(eval_size, n_queries)):

            start_doc, final_doc = eval_data.query_range(qid)

            if final_doc - start_doc < 2:
                continue

            cur_features = feat_matrix[start_doc:final_doc].clone(
            ).detach().float().to('cpu')
            cur_labels = labels[start_doc:final_doc].clone(
            ).detach().float().to('cpu')

            out = net(cur_features)

    ndcg = evl.evaluate(eval_data, scores)["ndcg"][0]
    new_scores = (1/(1 + np.exp(-scores)))
    new_scores *= 4
    err = evl.err(new_scores, labels)

    return ndcg, err


def stopping_condition(eval_ndcgs):
    """
    Determine early stopping condition, save model if it is the best yet.
    """
    recent_eval_ndcgs = eval_ndcgs[-10:]

    # determine trend in ndcg for the last number of epochs
    trend = (recent_eval_ndcgs[-1] -
             sum(recent_eval_ndcgs) / len(recent_eval_ndcgs))

    return np.sign(trend) == -1


def train(learning_rate=1e-4, architecture=[64, 32]):

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    train_data = data.train
    eval_data = data.validation
    test_data = data.test

    n_train_queries = train_data.num_queries()

    net = RankNet(architecture=architecture)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_losses = []
    eval_losses = []
    eval_ndcgs = []
    eval_errs = []

    for epoch in range(5):

        for qid in range(1, n_train_queries):

            net.train()
            optimizer.zero_grad()

            start_doc, final_doc = train_data.query_range(qid)

            if final_doc - start_doc < 2:
                continue

            features = torch.tensor(
                train_data.feature_matrix[start_doc:final_doc]).float().to('cpu')
            labels = torch.tensor(
                train_data.label_vector[start_doc:final_doc]).float().to('cpu')

            predictions = net.forward(features)

            lambdas = torch.zeros(predictions.shape)
            # Consider only pairs of labels for which S_ij = 1, as per papers suggestion.
            for i, u_i in enumerate(labels):
                lambda_i = 0
                for j, u_j in enumerate(labels):
                    if u_i > u_j:
                        lambda_ij = 1 * (0.5 * (1 - 1) - (1 / (1 +
                                                               torch.exp(1 * (predictions[i] - predictions[j])))))
                        lambda_i += lambda_ij
                lambdas[i] = lambda_i

            # create var to accumulate gradients for this query
            for p in net.parameters():
                p.inter_grad = torch.zeros(p.shape, requires_grad=False)

            # calculate gradients, accumulate them
            for i, doc in enumerate(features):
                optimizer.zero_grad()
                doc_out = net.forward(doc.view(1, 501))
                doc_out.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 1e3)

                with torch.no_grad():
                    for p in net.parameters():
                        p.inter_grad += lambdas[i] * p.grad

            # update gradients
            with torch.no_grad():
                for p in net.parameters():
                    new_p = p - learning_rate * p.inter_grad
                    p.copy_(new_p)

            if not qid % 100:

                eval_ndcg, eval_err = evaluate(
                    net, eval_data)
                eval_ndcgs.append(eval_ndcg)

                print('epoch: {}, query: {}/{} - evaluation ndcg: {}, evaluation err: {}'.format(epoch, qid,
                                                                                                 n_train_queries, eval_ndcg, eval_err))
                if stopping_condition(eval_ndcgs):
                    test_ndcg, test_err = evaluate(net, test_data)
                    print("Early stopping condition reached.")
                    print("NDCG: {}, ERR: {}".format(test_ndcg, test_err))
                    torch.save(net.state_dict(
                    ), 'saved_models/ranknet_spedup_{}_{}.pt'.format(learning_rate, str(architecture)))
                    return True


if __name__ == '__main__':

    # Perform a hyperparameter search

    # size_hidden = [500]
    # amount_hidden = 2
    # for learning_rate in [1e-5, 1e-4, 1e-3]:
    #     architecture = size_hidden*amount_hidden
    #     print('Training network with lr: {} and architecture: {}'.format(
    #         learning_rate, str(architecture)))
    #     train(learning_rate, architecture)

    # learning_rate = 1e-3
    # amount_hidden = 2
    # for size_hidden in [[50], [200], [500], [1000]]:
    #     architecture = size_hidden*amount_hidden
    #     print('Training network with lr: {} and architecture: {}'.format(
    #         learning_rate, str(architecture)))
    #     train(learning_rate, architecture)

    # size_hidden = [500]
    # learning_rate = 1e-3
    # for amount_hidden in [1, 2, 3, 4]:
    #     architecture = size_hidden*amount_hidden
    #     print('Training network with lr: {} and architecture: {}'.format(
    #         learning_rate, str(architecture)))
    #     train(learning_rate, architecture)

    # Train and evaluate the best model
    architecture = [500, 500]
    learning_rate = 0.00001
    print('Training network with lr: {} and architecture: {}'.format(
        learning_rate, str(architecture)))
    train(learning_rate, architecture)
