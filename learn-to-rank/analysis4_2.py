import torch
import dataset

import ipdb

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

class ListWiseRankNet(torch.nn.Module):

    def __init__(self, input_dim=501, architecture=[64, 32], output_dim=1):
        super(ListWiseRankNet, self).__init__()

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
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.sigmoid(x)
        return x * 4



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


loss = torch.nn.MSELoss().to(DEVICE)

data = dataset.get_dataset().get_data_folds()[0]
data.read_data()

data = data.test

X_cuda = torch.tensor(data.feature_matrix).float().to(DEVICE)
y_cuda = torch.tensor(data.label_vector).float().to(DEVICE)

X = torch.tensor(data.feature_matrix).float()
y = torch.tensor(data.label_vector).float()


point = PointwiseRanker(architecture=[150,150,150,150,150]).to(DEVICE)
point.load_state_dict(torch.load("./saved_models/pointwise_lameopslaanmanierversie.pt"))
point.eval()

out_point = point(X_cuda)

pair = RankNet(architecture=[500,500])
pair.load_state_dict(torch.load("./saved_models/ranknet_spedup_1e-05_[500, 500].pt"))
pair.eval()

out_pair = pair(X)

listWise = ListWiseRankNet(architecture=[200])
listWise.load_state_dict(torch.load("./saved_models/lambdarank_1e-05_[200]"))
listWise.eval()

out_list = listWise(X)

loss_point = loss(out_point.squeeze(), y_cuda)

loss_pair = loss(out_pair, y)

loss_list = loss(out_list, y)

# ipdb.set_trace()

print(f"\n\nloss for pointwise: {loss_point}\nloss for pairwise: {loss_pair}\nloss for listwise: {loss_list}\n\n")