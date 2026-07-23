import copy
import gc

import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch.nn import functional as F

from consts import N_EPOCHS, DEVICE, LEARNING_RATE


class Submodel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, p: float = 0.2):
        super().__init__()
        self._fc = nn.Sequential(
            nn.Linear(input_size, int(hidden_size / 2)),
            nn.BatchNorm1d(int(hidden_size / 2)),
            nn.Dropout(p=p),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_size / 2), hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=p),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self._fc(x)


class SPIDERMouse(nn.Module):
    def __init__(self, expression_size: int, prot_size: int, locations_size: int, graph_matrix: pd.DataFrame,
                 p: float = 0.3, hidden_size: int = 64):
        super().__init__()
        self.best_model = None
        self._loc_len = int(locations_size / 2)
        self._g_len = int(expression_size / 2)
        self._p_len = int(prot_size / 2)

        self._g_head = Submodel(input_size=self._g_len, hidden_size=hidden_size, p=p).to(DEVICE)

        self._l_head = Submodel(input_size=self._loc_len, hidden_size=hidden_size, p=p).to(DEVICE)

        self._p_head = Submodel(input_size=self._p_len, hidden_size=hidden_size, p=p).to(DEVICE)

        self._co_p_head = Submodel(input_size=1, hidden_size=hidden_size, p=p).to(DEVICE)

        self._co_loc_head = Submodel(input_size=1, hidden_size=hidden_size, p=p).to(DEVICE)

        # self._gat = gat_model
        self._gat1 = GATv2Conv(3 * hidden_size, hidden_size, heads=4, dropout=p,
                               add_self_loops=False).to(DEVICE)

        self._phi1 = nn.Sequential(
            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(2 * hidden_size),
            nn.Dropout(p=p),
            nn.LeakyReLU(),
        ).to(DEVICE)

        self._rho1 = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(2 * hidden_size),
            nn.Dropout(p=p / 2),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=p / 2),
            nn.LeakyReLU(),
        ).to(DEVICE)

        self._fc = nn.Sequential(
            nn.Linear(3 * hidden_size, int(hidden_size)),
            nn.BatchNorm1d(int(hidden_size)),
            # nn.Dropout(p=p / 2),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_size), int(hidden_size / 2)),
            nn.BatchNorm1d(int(hidden_size / 2)),
            # nn.Dropout(p=p /2),
            nn.LeakyReLU(),
            # nn.Linear(int(hidden_size / 2), int(hidden_size / 4)),
            # nn.BatchNorm1d(int(hidden_size / 2)),
            # # nn.Dropout(p=p / 2),
            # nn.LeakyReLU(),
            nn.Linear(int(hidden_size / 2), 1),
        ).to(DEVICE)

        self._sigmoid = nn.Sigmoid()
        self._ce_loss = nn.BCELoss()
        # self._recon_loss = nn.MSELoss()

        for p in self.parameters():
            try:
                nn.init.uniform_(p, -0.2, 0.2)
            except ValueError:
                if len(p.shape) == 1:
                    p.requires_grad = False
        self._graph_matrix = graph_matrix
        self._train_graph_matrix = graph_matrix
        self._val_graph_matrix = None
        self._test_graph_matrix = None
        self._graph_matrix.requires_grad = False

    def forward(self, x):
        interaction, edge_index = x
        n_edges = edge_index.shape[0]
        edge_index2 = torch.concat((edge_index, edge_index.permute(0, 1)), dim=0)
        gm_g = self._g_head(self.graph_matrix[:, : self._g_len])
        gm_p = self._p_head(self.graph_matrix[:, self._g_len: self._g_len + self._p_len])
        gm_l = self._l_head(self.graph_matrix[:, self._g_len + self._p_len: self._g_len + self._p_len + self._loc_len])
        gm = torch.concat((gm_g, gm_p, gm_l), axis=1)
        gm = F.dropout(gm, p=0.15)
        h1, (attn, _) = self._gat1(gm, edge_index2.T, return_attention_weights=True)
        preds1 = self._phi1(h1[edge_index[:, 0]]) + self._phi1(h1[edge_index[:, 1]])
        preds = self._rho1(preds1)
        inter_p = self._co_p_head(interaction[:, :1])
        inter_l = self._co_loc_head(interaction[:, 1:].permute(0, 1))
        preds = torch.concat([preds, inter_p, inter_l], dim=-1)
        preds = self._fc(preds)
        return self._sigmoid(preds.view(-1)), attn

    @property
    def graph_matrix(self):
        if self.training:
            return self._train_graph_matrix
        return self._test_graph_matrix

    @graph_matrix.setter
    def graph_matrix(self, gm):
        if self.training:
            self._train_graph_matrix = gm
        else:
            self._test_graph_matrix = gm

    def freeze_layers(self):
        self._gat1.requires_grad_(False)
        self._phi1.requires_grad_(False)
        self._g_head.requires_grad_(False)
        self._l_head.requires_grad_(False)
        self._p_head.requires_grad_(False)
        self._co_p_head.requires_grad_(False)
        self._co_loc_head.requires_grad_(False)

    def evaluate(self, dataloader, dataset):
        with torch.no_grad():
            self.eval()
            self.graph_matrix = dataset.graph_matrix
            y_preds, _ = self((dataset.interaction, dataset.edge_index))
            y_true = dataset.y.cpu().detach().numpy()
            y_scores = y_preds.cpu().detach().numpy()
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        self.train()
        return auc(recall, precision)

    def train_all(self, dataloaders, datasets, epochs: int = N_EPOCHS, learning_rate: float = LEARNING_RATE):
        self.train()
        optim = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        best_validation_accuracy = - float('inf')
        for epoch in range(epochs):
            total = 0
            running_loss = 0.0
            # for data in dataloaders['train']:
            optim.zero_grad()
            training_data, training_labels = (datasets['train'].interaction, datasets['train'].edge_index), datasets[
                'train'].y
            if training_data[0].shape[0] == 1:
                continue
            # Run forward pass and compute loss along the way.
            preds, attn = self.forward(training_data)
            loss = self._ce_loss(preds, training_labels)

            # Perform backpropagation
            loss.backward()
            # Update parameters
            optim.step()

            # Training stats
            total += 1
            running_loss += loss.item()

            # Evaluate and track improvements on the validation dataset
            training_accuracy = self.evaluate(dataloaders['train'], datasets['train'])
            validation_accuracy = self.evaluate(dataloaders['val'], datasets['val'])
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                self.best_model = copy.deepcopy(self.state_dict())
            epoch_loss = running_loss / total
            if not epoch % 50:
                print(
                    f"""Epoch: {epoch} Loss: {epoch_loss:.4f} Train acc: {training_accuracy} Val acc: {validation_accuracy}""")
        gc.collect()
        torch.cuda.empty_cache()
        self.load_state_dict(self.best_model)
