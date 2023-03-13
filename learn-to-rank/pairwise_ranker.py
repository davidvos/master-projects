import torch
import sys
import os

import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import random
import matplotlib.pyplot as plt

import itertools

from torch.autograd import Variable


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
    losses = []

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
            loss = pairwise_loss(out, cur_labels).detach().item()
            losses.append(loss)

    ndcg = evl.evaluate(eval_data, scores)["ndcg"][0]
    arr = np.mean(evl.evaluate(eval_data, scores)["relevant rank"][0])
    new_scores = (1/(1 + np.exp(-scores))) * 4
    err = evl.err(new_scores, labels)

    return ndcg, err, arr, np.mean(losses)


def pairwise_loss(predictions, labels):
    query_loss = torch.zeros(1)
    for i in range(len(labels)):
        for j in range(len(labels)):

            s_i, s_j = predictions[i], predictions[j]
            u_i, u_j = labels[i], labels[j]
            C = torch.log(1 + torch.exp(torch.sigmoid(s_i - s_j)))
            if u_i < u_j:
                C += torch.sigmoid(s_i - s_j)
            elif u_i == u_j:
                continue
            query_loss += C
    return query_loss / len(labels)


def stopping_condition(train_losses, eval_losses, eval_ndcgs, eps=1e-2):
    """
    Determine early stopping condition, save model if it is the best yet.
    """
    recent_train_losses = train_losses[-10:]
    recent_eval_losses = eval_losses[-10:]
    recent_eval_ndcgs = eval_ndcgs[-10:]

    # determine divergence between training and evaluation loss
    diff_loss = np.mean(recent_train_losses) - np.mean(recent_eval_losses)

    # normalize
    diff_loss /= max(recent_eval_losses)

    # determine trend in ndcg for the last number of epochs
    trend = (recent_eval_ndcgs[-1] -
             sum(recent_eval_ndcgs) / len(recent_eval_ndcgs))

    print("Loss difference: {}".format(diff_loss))

    return diff_loss > eps and np.sign(trend) == -1


def plot_ndcg_arr(eval_ndcgs, eval_arrs):
    x = [i*100 for i in range(len(eval_ndcgs))]
    y1 = eval_ndcgs
    y2 = eval_arrs

    plt.plot(x, y1, label="NDCG")
    plt.xlabel('Batch')
    # Set the y axis label of the current axis.
    plt.ylabel('NDCG')
    # Set a title of the current axes.
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()
    plt.plot(x, y2, label="ARR")
    plt.xlabel('Batch')
    # Set the y axis label of the current axis.
    plt.ylabel('ARR')
    # Set a title of the current axes.
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()


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
    eval_arrs = []
    eval_errs = []

    for epoch in range(5):

        for qid in range(0, n_train_queries):

            net.train()
            optimizer.zero_grad()

            start_doc, final_doc = train_data.query_range(qid)

            if final_doc - start_doc < 2:
                continue

            features = torch.tensor(
                train_data.feature_matrix[start_doc:final_doc]).float().to('cpu')
            labels = torch.tensor(
                train_data.label_vector[start_doc:final_doc]).float().to('cpu')
            query_out = net.forward(features)

            query_loss = pairwise_loss(query_out, labels)

            train_losses.append(query_loss.detach().item())

            if query_loss != 0:
                query_loss.backward()
                optimizer.step()

            if not qid % 100:

                eval_ndcg, eval_err, eval_arr, eval_loss = evaluate(
                    net, eval_data)
                eval_ndcgs.append(eval_ndcg)
                eval_arrs.append(eval_arr)
                eval_losses.append(eval_loss)

                print('epoch: {}, query: {}/{} - evaluation ndcg: {}, evaluation err: {}, evaluation arr: {}'.format(epoch, qid,
                                                                                                                     n_train_queries, eval_ndcg, eval_err, eval_arr))
                if len(eval_losses) > 10 and stopping_condition(train_losses, eval_losses, eval_ndcgs):
                    test_ndcg, test_err, _, _ = evaluate(net, test_data)
                    print("Early stopping condition reached.")
                    print("NDCG: {}, ERR: {}".format(test_ndcg, test_err))
                    torch.save(net.state_dict(
                    ), 'saved_models/ranknet_{}_{}.pt'.format(learning_rate, str(architecture)))
                    # Plot NDCG and ARR on evaluation set
                    # plot_ndcg_arr(eval_ndcgs, eval_arrs)
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
    architecture = [200]
    learning_rate = 0.001
    print('Training network with lr: {} and architecture: {}'.format(
        learning_rate, str(architecture)))
    train(learning_rate, architecture)
