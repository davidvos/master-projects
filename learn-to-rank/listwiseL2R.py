import torch
import sys
import os

import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import random

import matplotlib.pyplot as plt
import argparse

import itertools

from torch.autograd import Variable


class ListWiseRankNet(torch.nn.Module):

    def __init__(self, input_dim=501, architecture=[64, 32], output_dim=1):
        super(RankNet, self).__init__()

        self.layers = torch.nn.ModuleList()

        prev_layer = input_dim
        for layer in architecture:
            self.layers.append(torch.nn.Linear(prev_layer, layer))
            self.layers.append(torch.nn.LeakyReLU(0.2))
            prev_layer = layer
        self.layers.append(torch.nn.Linear(prev_layer, output_dim))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.sigmoid(x) * 4
        return x


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
# torch.autograd.set_detect_anomaly(True)


def listwise_loss(predictions, labels, IRM="ndcg", sigma=3.147):
    orig_irm_score = evaluate_IRM(predictions, labels, 0, IRM)
    tot_C = torch.tensor([0]).float().to(DEVICE)
    global_count = torch.tensor(0).to(DEVICE)
    sigma = torch.tensor(sigma).to(DEVICE)

    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            global_count += 1
            s_i, s_j = predictions[i], predictions[j]
            u_i, u_j = labels[i], labels[j]

            if u_i <= u_j:
                S = 0 if u_i == u_j else -1
            elif u_i > u_j:
                S = 1

            C = .5 * (1 - S) * sigma * (s_i - s_j) + \
                torch.log(1 + torch.exp(sigma * (s_i - s_j)))
            tot_C += C * abs(orig_irm_score - torch.tensor(swap(i,
                                                                j, predictions, labels, IRM)).to(DEVICE))

    # print(f"len labels: {len(labels)} lamda: {(tot_C/global_count).to(DEVICE)}")
    return (tot_C/global_count).to(DEVICE)


def evaluate_IRM(predictions, labels, p=0, IRM="ndcg"):
    IRM = IRM.lower()
    if IRM.lower() == "ndcg":
        if DEVICE == torch.device("cuda:0"):
            pred, lab = torch.argsort(
                predictions.view(-1)).cpu().numpy(), torch.argsort(labels).cpu().numpy()
        else:
            pred, lab = torch.argsort(
                predictions.view(-1)).numpy(), torch.argsort(labels).numpy()
        return evl.ndcg_at_k(pred, lab, p)
    elif IRM == "err":
        return ERR(predictions, labels, p)


def swap(i, j, pred, lab, IRM):
    val = pred[j]
    pred[j] = pred[i]
    pred[i] = val
    return evaluate_IRM(pred, lab, 0, IRM)


def evaluate(net, eval_data, IRM="ndcg", sigma=3.147):

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
            loss = listwise_loss(out, cur_labels, IRM, sigma).detach().item()
            losses.append(loss)

    ndcg = evl.evaluate(eval_data, scores)["ndcg"][0]
    err = evl.err(scores, labels)

    return ndcg, err, losses


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


def train(learning_rate=1e-4, architecture=[64, 32], epochs=5, IRM="ndcg", sigma=3.147):

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    train_data = data.train
    eval_data = data.validation
    test_data = data.test

    n_train_queries = train_data.num_queries()

    net = RankNet(architecture=architecture)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_ndcgs, eval_ndcgs = [], []
    best_eval_ndcg = float("inf")

    best_score = -np.inf
    best_count = 0

    train_losses = []
    train_losses_2 = []
    eval_ndcgs_list = []
    x_axis = []

    eval_ndcgs = []
    eval_errs = []
    for epoch in range(epochs):

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

            query_loss = listwise_loss(query_out, labels, IRM, sigma)

            train_losses.append(query_loss.detach().item())
            if query_loss != 0:
                query_loss.backward()
                optimizer.step()

            if not qid % 5:
                eval_ndcg, _, _ = evaluate(
                    net, eval_data, IRM)

                eval_ndcgs_list.append(eval_ndcg)
                x_axis.append(qid)

            if not qid % 100:

                eval_ndcg, eval_err, eval_losses = evaluate(
                    net, eval_data, IRM)
                eval_ndcgs.append(eval_ndcg)

                print('epoch: {}, query: {}/{} - evaluation ndcg: {}, evaluation err: {}'.format(epoch, qid,
                                                                                                 n_train_queries, eval_ndcg, eval_err))
                if len(eval_ndcgs) > 10 and stopping_condition(train_losses, eval_losses, eval_ndcgs):
                    test_ndcg, test_err, _ = evaluate(net, test_data, IRM)
                    print("Early stopping condition reached.")
                    print("NDCG: {}, ERR: {}".format(test_ndcg, test_err))
                    torch.save(net.state_dict(
                    ), 'saved_models/lambdarank_{}_{}'.format(learning_rate, str(architecture)))


                    plt.plot(x_axis, eval_ndcgs_list, label="evaluation ndcg")
                    plt.savefig("losses.jpg")
                    plt.show()

                    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", help="learning rate", default=0.001, type=float)
    parser.add_argument("-hidden_layers", help="architecture of the network",
                        nargs='*', type=int, default=[200])
    parser.add_argument(
        "-epochs", help="no epochs to train for", type=int, default=100)

    parser.add_argument("-IRM", type=str, default="ndcg", choices={"ndcg", "err"}, 
                            help="sets the IRM used for the lambdarank computation")
    parser.add_argument("-sigma", type=float, default=3.147, 
                            help="defines the sigma value used in the C computation")

    parser.add_argument("--hyperparam_search",
                        action="store_true", default=False)

    args = parser.parse_args()


    if args.hyperparam_search:
        size_hidden = [20]
        amount_hidden = 1
        for learning_rate in [1e-5, 1e-4, 1e-3, 1e-2]:
            architecture = size_hidden*amount_hidden
            print('Training network with lr: {} and architecture: {}'.format(
                learning_rate, str(architecture)))
            train(learning_rate, architecture)

        learning_rate = 1e-5
        amount_hidden = 1
        for size_hidden in [[20], [50], [100], [200]]:
            architecture = size_hidden*amount_hidden
            print('Training network with lr: {} and architecture: {}'.format(
                learning_rate, str(architecture)))
            train(learning_rate, architecture)

        size_hidden = [20]
        learning_rate = 1e-5
        for amount_hidden in [1, 2, 3, 4]:
            architecture = size_hidden*amount_hidden
            print('Training network with lr: {} and architecture: {}'.format(
                learning_rate, str(architecture)))
            train(learning_rate, architecture)
    else:
        learning_rate = 1e-3
        architecture = [200]
        print('Training network with lr: {} and architecture: {}'.format(
            learning_rate, str(architecture)))
        train(args.lr, args.hidden_layers, args.epochs, args.IRM, args.sigma)
