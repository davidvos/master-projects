# Information Retrieval Assignment 3 UvA

This repository contains three different learning to rank implementations; Pointwise L2R, RankNet and LambdaRank. 

## Pointwise L2R

The Pointwise L2R model is implemented in `pointwise_ranker.py`. The model can be trained, saved and evaluated by calling this file from the command line.

Optional parameters are: `-lr` for the learning rate, a float with a default value of 0.001. `-hidden_layers` which takes a list of integers denoting the dimensionality of the hidden layers to train the model with. `-epochs`, an integer denoting the maximum number of epochs to train for, note that this is a theorethical maximum and training may halt earlier due to the implemented early stopping condition. `-save_path`, a string denoting the filepath at which to save the trained model, with default value `./saved_models/pointwise_ranker.pt`. `-batch_size`, an integer denoting the batch size with which to train the  model. `--load_model`, a flag which, when enabled, will attempt to load the model from the save path.

Lastly `--hyperparam_search`, when this flag is enabled the script will perform a grid search over optimal hyperparameters: for the learning rate it will search through the values [0.0001, 0.001, 0.01, 0.1], for the number of hidden layers it will search through the values [1, 3, 5, 10, 50], and for the dimensionality of the hidden layers it will search through the values [10, 50, 100, 150, 500]. When one hyperparameter is searched for, the others are kept at fixed values to reduce the search complexity. These default values for learning rate, number of hidden layers and hidden layer dimensionality are 0.001, 3 and 100 respectively.

If hyperparemeter search is not performed, the model will train with the user-specified hyperparameters. After training has completed, the training loss, evaluation loss and evaluation NDCG are plotted. Finally evaluation results over the test set are printed and the distribution of predicted labels is plotted in a histogram versus the distribution of true labels.

For example the following call will train a network with three hidden layers with dimensionalities 100, 500 and 100 respectively, with a learning rate of 0.01:

```bash
python pointwise_ranker.py -lr 0.01 -hidden_layers 100 500 100
```

This outputs:

```bash
ModuleList(
  (0): Linear(in_features=501, out_features=100, bias=True)
  (1): LeakyReLU(negative_slope=0.2)
  (2): Linear(in_features=100, out_features=500, bias=True)
  (3): LeakyReLU(negative_slope=0.2)
  (4): Linear(in_features=500, out_features=100, bias=True)
  (5): LeakyReLU(negative_slope=0.2)
  (6): Linear(in_features=100, out_features=1, bias=True)
)
EPOCH: 0
iteration: 50 - loss: 0.6934
EPOCH: 0 ITERATION : 50 - eval ndcg 0.8176 - eval loss 0.7609
iteration: 100 - loss: 0.7071
EPOCH: 0 ITERATION : 100 - eval ndcg 0.8333 - eval loss 0.7160
iteration: 150 - loss: 0.6592
EPOCH: 0 ITERATION : 150 - eval ndcg 0.8348 - eval loss 0.7021
...
```


## Listwise L2R

The Listwise L2R model is implemented in `listwise_ranker.py`. The model can be trained, saved and evaluated by calling this file from the command line.

Optional parameters are: `-lr` for the learning rate, a float with a default value of 0.001. `-hidden_layers` which takes a list of integers denoting the dimensionality of the hidden layers to train the model with. `-epochs`, an integer denoting the maximum number of epochs to train for, note that this is a theorethical maximum and training may halt earlier due to the implemented early stopping condition. `-IRM`, a string being either 'ndcg' or 'err' denoting which ranking measure should be used in computation. `-sigma`, a float denoting the value of sigma in the equation (3) of the homework pdf.

Lastly `--hyperparam_search`, when this flag is enabled the script will perform a grid search over optimal hyperparameters: for the learning rate it will search through the values [0,00001, 0.0001, 0.001, 0.01], for the number of hidden layers it will search through the values [1, 3, 5, 10, 50], and for the dimensionality of the hidden layers it will search through the values [10, 50, 100, 150, 500]. When one hyperparameter is searched for, the others are kept at fixed values to reduce the search complexity. These default values for learning rate, number of hidden layers and hidden layer dimensionality are 0.00001, 1 and 20 respectively.

If hyperparemeter search is not performed, the model will train with the user-specified hyperparameters. After training has completed, evaluation results over the test set are printed.

For example the following call will train a network with three hidden layers with dimensionalities 256, 128 and 64 respectively, with a learning rate of 0.001:

```bash
python listwise_ranker.py -lr 0.001 -hidden_layers 200
```
will result in:

```bash
Training network with lr: 0.001 and architecture: [200]
epoch: 0, query: 100/19943 - evaluation ndcg: 0.789845690834123, evaluation err: 0.905455870919737
epoch: 0, query: 200/19943 - evaluation ndcg: 0.7929159653277327, evaluation err: 0.9364041063647783
```

## Built With

* [Pytorch](https://pytorch.org/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
