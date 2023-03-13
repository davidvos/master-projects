import torch
import sys

import dataset
import ranking as rnk
import evaluate as evl
import numpy as np

import matplotlib.pyplot as plt

import argparse
import json

from pprint import pprint
from collections import defaultdict

import os

#DEVICE = "cpu"
DEVICE = "cuda"


class PointwiseRanker(torch.nn.Module):

    def __init__(self, input_dim=501, architecture=[100], output_dim=1):
        super(PointwiseRanker, self).__init__()

        self.layers = torch.nn.ModuleList()

        prev_layer = input_dim
        for layer in architecture:
            self.layers.append(torch.nn.Linear(prev_layer, layer))
            self.layers.append(torch.nn.LeakyReLU(0.2))
            prev_layer = layer
        self.layers.append(torch.nn.Linear(prev_layer, output_dim))

        print(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def stopping_condition(train_losses, eval_losses, eval_ndcgs, eps=1e-1):
    """
    Determine early stopping condition, save model if it is the best yet.
    """
    recent_train_losses = train_losses[-5:]
    recent_eval_losses = eval_losses[-5:]
    recent_eval_ndcgs = eval_ndcgs[-5:]
    diff_loss = 0

    # determine divergence between training and evaluation loss
    for train_loss, eval_loss in zip(recent_train_losses, recent_eval_losses):
        diff_loss += abs(eval_loss - train_loss)

    diff_loss /= len(recent_eval_losses)

    # normalize
    diff_loss /= max(recent_eval_losses)

    # determine trend in ndcg for the last number of epochs
    trend = (recent_eval_ndcgs[-1] -
             sum(recent_eval_ndcgs) / len(recent_eval_ndcgs))

    print("Loss difference: {}".format(diff_loss))

    return diff_loss > eps and np.sign(trend) == -1


def compute_distributions(feat_matrix, labels, model, batch_size=1024):
    """
    Visualise the distribution of true ranking grades versus the distribution of 
    computed ranking grades.
    """
    model.eval()

    scores = np.zeros(labels.shape)
    loss = 0

    for iteration, batch_start in enumerate(list(range(0, len(feat_matrix), batch_size))):
        batch_inputs = feat_matrix[batch_start: batch_start + batch_size]
        batch_labels = labels[batch_start: batch_start + batch_size]

        out = model(batch_inputs)
        loss += loss_fn(out.squeeze(), batch_labels)
        scores[batch_start: batch_start +
               batch_size] = out.detach().squeeze().cpu().numpy()

    loss /= (iteration + 1)

    plt.hist(scores, color="orange", label="predicted ranking scores",
             alpha=0.5, bins=[-1, 0, 1, 2, 3, 4, 5, 6])
    plt.hist(labels.cpu().numpy(), color="blue", label="true ranking scores",
             alpha=0.5, bins=[-1, 0, 1, 2, 3, 4, 5, 6])
    plt.legend()
    # plt.show()
    plt.savefig("./images/dist_pointwise.png")


def eval_pointwise_model(split, feat_matrix, labels, loss_fn, model, batch_size=1024, full_eval=False):
    """
    Evaluate the pointwise model on the validation dataset.
    """

    # one score for every document (1d vector in ordering of the dataset)
    scores = np.zeros(labels.shape)
    loss = 0

    with torch.no_grad():
        for iteration, batch_start in enumerate(list(range(0, len(feat_matrix), batch_size))):
            batch_inputs = feat_matrix[batch_start: batch_start + batch_size]
            batch_labels = labels[batch_start: batch_start + batch_size]

            out = model(batch_inputs)
            loss += loss_fn(out.squeeze(), batch_labels)
            scores[batch_start: batch_start +
                   batch_size] = out.detach().squeeze().cpu().numpy()

    if full_eval:
        return evl.evaluate(split, scores)

    ndcg = evl.evaluate(split, scores)["ndcg"][0]
    loss = loss / (iteration + 1)

    return ndcg, loss


def train_model(model, feat_matrix, eval_feat_matrix, labels, eval_labels, loss_fn, epochs, batch_size, save_path, save=True):

    best_eval_loss = float("inf")

    stop = False

    #indices_tracker = list(range(feat_matrix.shape[0]))
    train_losses, train_ndcgs, eval_losses, eval_ndcgs = [], [], [], []
    for epoch in range(epochs):

        model.train()
        print("EPOCH: {}".format(epoch))

        # randomly shuffle dataset.
        shuffled_ind = np.random.permutation(feat_matrix.shape[0])
        feat_matrix = feat_matrix[shuffled_ind]
        labels = labels[shuffled_ind]
        #indices_tracker = indices_tracker[shuffled_ind]

        for iteration, batch_start in enumerate(list(range(0, len(feat_matrix), batch_size))):

            optim.zero_grad()

            batch_inputs = feat_matrix[batch_start: batch_start + batch_size]
            batch_labels = labels[batch_start: batch_start + batch_size]

            out = model(batch_inputs)

            loss = loss_fn(out.squeeze(), batch_labels)

            if iteration and not iteration % 50:
                print("iteration: {} - loss: {:.4f}".format(iteration, loss))
                train_ndcg, train_loss = eval_pointwise_model(
                    data.train, feat_matrix, labels, loss_fn, model)
                eval_ndcg, eval_loss = eval_pointwise_model(
                    data.validation, eval_feat_matrix, eval_labels, loss_fn, model)

                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
                # train_ndcgs.append(train_ndcg)
                eval_ndcgs.append(eval_ndcg)

                print("EPOCH: {} ITERATION : {} - eval ndcg {:.4f} - eval loss {:.4f}".format(
                    epoch, iteration, eval_ndcg, eval_loss))

                if stopping_condition(train_losses, eval_losses, eval_ndcgs):
                    print("Early stopping condition reached.")
                    stop = True
                    break

                if save and eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    torch.save(model, save_path)

            loss.backward()
            optim.step()

        if stop:
            break

    return train_losses, train_ndcgs, eval_losses, eval_ndcgs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", help="learning rate", default=0.001, type=float)
    parser.add_argument("-hidden_layers", help="architecture of the network",
                        nargs='*', type=int, default=[100])
    parser.add_argument(
        "-epochs", help="no epochs to train for", type=int, default=100)
    parser.add_argument("-save_path", type=str,
                        default="./saved_models/pointwise_ranker.pt")
    parser.add_argument("-batch_size", type=int, default=1024)

    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--hyperparam_search",
                        action="store_true", default=False)

    args = parser.parse_args()

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    feat_matrix = torch.FloatTensor(data.train.feature_matrix).to(DEVICE)
    eval_feat_matrix = torch.FloatTensor(
        data.validation.feature_matrix).to(DEVICE)
    labels = torch.FloatTensor(data.train.label_vector).to(DEVICE)
    eval_labels = torch.FloatTensor(data.validation.label_vector).to(DEVICE)

    loss_fn = torch.nn.MSELoss().to(DEVICE)

    if args.hyperparam_search:

        print("Starting optimal hyperparameter search")

        best_hyperparams = defaultdict(lambda: {})

        def_lr = 0.001
        def_num_hidden = 3
        def_hidden_size = 100

        print("Default hyperparameters are: lr - {}, num_hidden - {}, hidden_size - {}".format(
            def_lr, def_num_hidden, def_hidden_size))

        print("Search over learning rates...")
        best_lr = None
        best_lr_ndcg = 0
        for lr in [0.0001, 0.001, 0.01, 0.1]:

            print("Learning rate: {}".format(lr))

            architecture = [def_hidden_size for _ in range(def_num_hidden)]

            model = PointwiseRanker(architecture=architecture).to(DEVICE)
            optim = torch.optim.Adam(model.parameters(), lr=lr)

            train_losses, train_ndcgs, eval_losses, eval_ndcgs = train_model(
                model, feat_matrix, eval_feat_matrix, labels, eval_labels, loss_fn, args.epochs, args.batch_size, args.save_path)

            best_hyperparams["lr"][str(lr)] = max(eval_ndcgs)

            if max(eval_ndcgs) > best_lr_ndcg:
                print("new best: {} {}".format(max(eval_ndcgs), lr))
                best_lr_ndcg = max(eval_ndcgs)
                best_lr = lr

        print("Search over number of hidden layers...")
        best_num_hidden = None
        best_nh_ndcg = 0
        for num_hidden in [1, 3, 5, 10, 50]:

            print("Number of hidden layers: {}".format(num_hidden))

            architecture = [def_hidden_size for _ in range(num_hidden)]

            model = PointwiseRanker(architecture=architecture).to(DEVICE)
            optim = torch.optim.Adam(model.parameters(), lr=def_lr)

            train_losses, train_ndcgs, eval_losses, eval_ndcgs = train_model(
                model, feat_matrix, eval_feat_matrix, labels, eval_labels, loss_fn, args.epochs, args.batch_size, args.save_path)

            best_hyperparams["num_hidden"][str(num_hidden)] = max(eval_ndcgs)

            if max(eval_ndcgs) > best_nh_ndcg:
                print("new best: {} {}".format(max(eval_ndcgs), num_hidden))
                best_nh_ndcg = max(eval_ndcgs)
                best_num_hidden = num_hidden

        print("Search over hidden layer sizes...")
        best_hl_size = None
        best_hl_ndcg = 0
        for hidden_size in [10, 50, 100, 150, 500]:

            print("Hidden layer size: {}".format(hidden_size))

            architecture = [hidden_size for _ in range(def_num_hidden)]

            model = PointwiseRanker(architecture=architecture).to(DEVICE)
            optim = torch.optim.Adam(model.parameters(), lr=def_lr)

            train_losses, train_ndcgs, eval_losses, eval_ndcgs = train_model(
                model, feat_matrix, eval_feat_matrix, labels, eval_labels, loss_fn, args.epochs, args.batch_size, args.save_path)

            best_hyperparams["hidden_size"][str(hidden_size)] = max(eval_ndcgs)

            if max(eval_ndcgs) > best_hl_ndcg:
                print("new best: {} {}".format(max(eval_ndcgs), hidden_size))
                best_hl_ndcg = max(eval_ndcgs)
                best_hl_size = hidden_size

        print("BEST HYPERPARAMETERS: lr {} hidden size {} num hidden {}".format(
            best_lr, best_hl_size, best_num_hidden))

        pprint(best_hyperparams)

        os.makedirs('./results', exist_ok=True)
        with open('./results/hyperparam_search_results.json', 'w+') as outfile:
            json.dump(best_hyperparams, outfile)

    else:

        if not args.load_model:
            model = PointwiseRanker(architecture=args.hidden_layers).to(DEVICE)
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)

            train_losses, train_ndcgs, eval_losses, eval_ndcgs = train_model(
                model, feat_matrix, eval_feat_matrix, labels, eval_labels, loss_fn, args.epochs, args.batch_size, args.save_path)

            plt.plot(train_losses, linestyle='--',
                     label="train loss", color="blue")
            plt.plot(eval_losses, label="eval loss", color="blue")
            #plt.plot(train_ndcgs, linestyle='--', label="train ndcg", color="red")
            plt.plot(eval_ndcgs, label="eval ndcg", color="red")
            plt.legend()
            # plt.show()
            plt.savefig("./images/loss_ndcg_pointwise.png")

        del feat_matrix
        del eval_feat_matrix
        del labels
        del eval_labels

        test_feat_matrix = torch.FloatTensor(
            data.test.feature_matrix).to(DEVICE)
        test_labels = torch.FloatTensor(data.test.label_vector).to(DEVICE)
        model = torch.load(args.save_path).to(DEVICE)

        evaluation = eval_pointwise_model(
            data.test, test_feat_matrix, test_labels, loss_fn, model, full_eval=True)

        os.makedirs('./results', exist_ok=True)
        with open('./results/pointwise_evaluation.json', 'w+') as outfile:
            json.dump(evaluation, outfile)

        pprint(evaluation)

        compute_distributions(test_feat_matrix, test_labels, model)
