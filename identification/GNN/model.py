import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GraphConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set, MulAggregation
from torch_geometric.nn import aggr
from torch_geometric.nn import GraphNorm
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_t, weights=None):
        loss = (y - y_t) ** 2
        if weights is not None:
            loss *= weights.expand_as(loss)
        return torch.mean(loss)


class GNN(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # self.fc2 = nn.Linear(200, 200)
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            in_dim = input_dim if layer == 0 else emb_dim
            if gnn_type == "gin":
                # self.gnns.append(GINConv(nn.Sequential(nn.Linear(in_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                #                                        nn.Linear(emb_dim, emb_dim))))
                self.gnns.append(GINConv(nn.Sequential(nn.Linear(in_dim, emb_dim), GraphNorm(emb_dim), nn.ReLU(),
                                                       nn.Linear(emb_dim, emb_dim), nn.ReLU())))
            elif gnn_type == "gcn":
                self.gnns.append(GraphConv(in_dim, emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(in_dim, emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv(in_dim, emb_dim))
            else:
                raise ValueError("Invalid GNN type.")

    def forward(self, x, edge_index, mut_res_idx, edge_attr=None):
        h_list = [x]
        mut_site = []
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # if layer == self.num_layer - 1:
            #     # remove relu from the last layer
            #     h = F.dropout(h, self.drop_ratio, training=self.training)
            # else:
            #     h = F.dropout(F.relu(h), self.drop_ratio, training=self.training) # F.relu()
            h_list.append(h)
            # if len(h_list) == 2:
            #     previous_mut_site_feature = h_list[-2][mut_res_idx]
            #     current_mut_site_feature = h_list[-1][mut_res_idx]
            #     # print(previous_mut_site_feature.shape, current_mut_site_feature.shape)
            #     h_feature = self.fc1(previous_mut_site_feature)
            #     h_list[-1][mut_res_idx] = h_feature + current_mut_site_feature
            # if len(h_list) == 3:
            #     previous_mut_site_feature = h_list[-2][mut_res_idx].squeeze(0)
            #     current_mut_site_feature = h_list[-1][mut_res_idx].squeeze(0)

            #     h_feature = self.fc2(previous_mut_site_feature) + current_mut_site_feature
            #     h_list[-1][mut_res_idx] = h_feature.unsqueeze(0)
            # mut_site.append()
        # print(len(h_list))
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)
        # print('node_rep', node_representation.shape)
        return h_list[-1]


# orthogonal initialization
def init_gru_orth(model, gain=1):
    model.reset_parameters()
    # orthogonal initialization of gru weights
    for _, hh, _, _ in model.all_weights:
        for i in range(0, hh.size(0), model.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + model.hidden_size], gain=gain)


def init_lstm_orth(model, gain=1):
    init_gru_orth(model, gain)

    # positive forget gate bias (Jozefowicz es at. 2015)
    for _, _, ih_b, hh_b in model.all_weights:
        l = len(ih_b)
        ih_b[l // 4: l // 2].data.fill_(1.0)
        hh_b[l // 4: l // 2].data.fill_(1.0)


class GraphGNN(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, out_dim, JK="last", drop_ratio=0, graph_pooling="attention",
                 gnn_type="gat", concat_type='lstm', fds=False, feature_level='both', contrast_curri=False) -> object:
        super(GraphGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.concat_type = concat_type
        self.feature_level = feature_level
        self.contrast_curri = contrast_curri

        self.seq_encoder = nn.Linear(50, self.emb_dim)
        self.graph_pool = nn.Linear(self.emb_dim, 1)
        self.global_local_att0 = nn.Linear(40, 10)

        self.fc1 = nn.Sequential(nn.Linear(40, self.emb_dim), nn.LeakyReLU(0.1), nn.Linear(self.emb_dim, 1))
        self.fc2 = nn.Sequential(nn.Linear(40, self.emb_dim), nn.LeakyReLU(0.1), nn.Linear(self.emb_dim, 1))
        self.fc3 = nn.Sequential(nn.Linear(40, self.emb_dim), nn.LeakyReLU(0.1), nn.Linear(self.emb_dim, 1))
        self.lstm_graph = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1, bidirectional=True, batch_first=True)


        if self.concat_type == 'lstm':
            # self.lstm_node = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1)
            self.lstm_graph = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1)
            # init_lstm_orth(self.lstm_node)
            self.fc = nn.Linear(self.emb_dim, self.out_dim)

        elif self.concat_type == 'bilstm':
            self.lstm_graph = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1, bidirectional=True)
            init_lstm_orth(self.lstm_graph)
            self.fc = nn.Linear(2*self.emb_dim, self.out_dim)

        elif self.concat_type == 'gru':
            self.lstm = nn.GRU(input_size=300, hidden_size=300, num_layers=1)
            init_gru_orth(self.lstm)
            self.fc = nn.Linear(2 * self.emb_dim, self.out_dim)

        else:

            if self.feature_level == 'global-local':
                self.fc = nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio), nn.Linear(self.emb_dim, self.out_dim))
                # self.fc = nn.Linear(self.emb_dim, self.out_dim)
                # self.fc = nn.Sequential(
                #     nn.Linear(2000, 5*self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                #     nn.Linear(5*self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),  nn.Linear(self.emb_dim, self.out_dim))
            else:
                self.fc = nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio), nn.Linear(self.emb_dim, self.out_dim))

        if fds:
            self.dir = True
        else:
            self.dir = False

        self.gnn = GNN(num_layer, input_dim, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "mul":
            self.pool = MulAggregation()
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        elif graph_pooling == "lstm":
            self.pool = aggr.LSTMAggregation(emb_dim, emb_dim)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward_once(self, x, edge_index, batch):
        node_representation = self.gnn(x, edge_index, 0)
        graph_rep = self.pool(node_representation, batch)
        return graph_rep

    def forward(self, data, epoch=0):

        graph_rep_be = self.forward_once(data.x_s, data.edge_index_s, data.x_s_batch)

        x = self.fc(graph_rep_be)
        # print(x.shape)
        return x
        # x1 = self.fc1(x)
        # x2 = self.fc2(x)
        # x3 = self.fc3(x)
        # return x1, x2, x3

