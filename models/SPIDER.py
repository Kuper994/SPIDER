import copy
import gc
import torch
import pandas as pd
from torch import nn
# from utils import DEVICE
from torch.utils.data import Dataset
from torch_geometric.nn import GATv2Conv
from torch.nn import functional as F
from sklearn.metrics import precision_recall_curve, auc

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpiderSubmodel(nn.Module):
  def __init__(self, input_size: int, hidden_size: int = 64, p: float = 0.2):
    super(SpiderSubmodel, self).__init__()
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


class SPIDER(nn.Module):
    def __init__(self, expression_size: int, prot_size: int, locations_size: int, graph_matrix: pd.DataFrame,
                 second_input_size: int, params = None,
                 p: float = 0.4, hidden_size: int = 64):
        super(SPIDER, self).__init__()
        self.best_model = None
        genes, prots, co_prots, locs, co_locs, methods = params or [True] * 6
        self._loc_len = int(locations_size / 2) if locs else 0
        self._g_len = int(expression_size / 2) if genes else 0
        self._p_len = int(prot_size / 2) if prots else 0
        self._m_len = second_input_size - int(co_locs) - int(co_prots) if methods else 0

        if genes:
          self._g_head = SpiderSubmodel(input_size=self._g_len, hidden_size=hidden_size, p=p).to(DEVICE)
        if locs:
          self._l_head = SpiderSubmodel(input_size=self._loc_len, hidden_size=hidden_size, p=p).to(DEVICE)
        if prots:
          self._p_head = SpiderSubmodel(input_size=self._p_len, hidden_size=hidden_size, p=p).to(DEVICE)
        if co_prots:
          self._co_p_head = SpiderSubmodel(input_size=1, hidden_size=hidden_size, p=p).to(DEVICE)
        if co_locs:
          self._co_loc_head = SpiderSubmodel(input_size=1, hidden_size=hidden_size, p=p).to(DEVICE)
        if methods:
          self._methods_head = SpiderSubmodel(input_size=self._m_len, hidden_size=hidden_size, p=p).to(DEVICE)

        num_comp = int(genes) + int(locs) + int(prots)

        if num_comp:
          self._gat1 = GATv2Conv(num_comp * hidden_size, hidden_size, heads=4, dropout=p).to(DEVICE)
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

        num_features = int(any([genes, prots, locs])) + int(co_prots) + int(co_locs) + int(methods)
        self._fc = nn.Sequential(
            nn.Linear(num_features * hidden_size, int(hidden_size)),
            nn.BatchNorm1d(int(hidden_size)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_size), int(hidden_size / 2)),
            nn.BatchNorm1d(int(hidden_size / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_size / 2), 1),
        ).to(DEVICE)

        self._sigmoid = nn.Sigmoid()
        self._ce_loss = nn.BCELoss()

        for p in self.parameters():
          try:
              nn.init.xavier_normal_(p)
          except ValueError:
              if len(p.shape) == 1:
                p.requires_grad = False
        self._graph_matrix = graph_matrix
        self._train_graph_matrix = graph_matrix
        self._params = params or [True] * 6
        self._val_graph_matrix = None
        self._test_graph_matrix = None

    def forward(self, x, return_gm=False):
        genes, prots, co_prots, locs, co_locs, methods = self._params
        interaction, edge_index = x
        n_edges = edge_index.shape[0]
        edge_index2 = torch.concat((edge_index, edge_index.permute(0, 1)), dim=0)
        to_concat = []
        if genes:
          gm_g = self._g_head(self.graph_matrix[:,: self._g_len])
          to_concat.append(gm_g)
        if prots:
          gm_p = self._p_head(self.graph_matrix[:, self._g_len: self._g_len + self._p_len])
          to_concat.append(gm_p)
        if locs:
          gm_l = self._l_head(self.graph_matrix[:, self._g_len + self._p_len: self._g_len + self._p_len + self._loc_len])
          to_concat.append(gm_l)
        h1 = None
        attn = None
        if to_concat:
          gm = torch.concat(to_concat, axis=1)
          gm = F.dropout(gm, p=0.15)
          h1, (_, attn) = self._gat1(gm, edge_index2.T, return_attention_weights=True)
          # h1 = F.dropout(h1, p=0.15)
          preds1 = self._phi1(h1[edge_index[:, 0]]) + self._phi1(h1[edge_index[:, 1]])
          preds = self._rho1(preds1)
          to_concat = [preds]

        i0 = int(co_prots)
        if co_prots:
          inter_p = self._co_p_head(interaction[:, :int(co_prots)])
          to_concat.append(inter_p)
        if co_locs:
          inter_l = self._co_loc_head(interaction[:, int(co_prots): int(co_prots) + int(co_locs)])
          to_concat.append(inter_l)
        if methods:
          inter_m = self._methods_head(interaction[:, int(co_prots) + int(co_locs):])
          to_concat.append(inter_m)

        preds = torch.concat(to_concat, dim=-1)
        preds = self._fc(preds)
        if return_gm:
          return self._sigmoid(preds.view(-1)), attn, h1
        else:
          return self._sigmoid(preds.view(-1))

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
      # self._rho1.requires_grad = False
      self._g_head.requires_grad_(False)
      self._l_head.requires_grad_(False)
      self._co_loc_head.requires_grad_(False)
      if self._p_expression:
        self._p_head.requires_grad_(False)
      if self._co_abundance:
        self._co_p_head.requires_grad_(False)
      self._methods_head.requires_grad_(False)

    def evaluate(self, dataset):
        with torch.no_grad():
          self.eval()
          self.graph_matrix = dataset.graph_matrix
          y_preds = self((dataset.interaction, dataset.edge_index))
          y_true = dataset.y.cpu().detach().numpy()
          y_scores = y_preds.cpu().detach().numpy()
          precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        self.train()
        return auc(recall, precision)

    def train_all(self, datasets, epochs: int = 15, learning_rate: float = 0.001):
        self.train()
        optim = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        best_validation_accuracy = - float('inf')
        with torch.no_grad():
            weights = torch.ones_like(datasets['train'].y).to(DEVICE)
            weights[datasets['train'].y == 1] = 2 * datasets['train'].y.shape[0] / datasets['train'].y.sum(0)
        criterion = nn.BCELoss(weight=weights)
        for epoch in range(epochs):
            total = 0
            running_loss = 0.0
            # for data in dataloaders['train']:
            optim.zero_grad()
            training_data, training_labels = (datasets['train'].interaction, datasets['train'].edge_index), datasets['train'].y

            preds = self.forward(training_data)
            loss = criterion(preds, training_labels)

            # Perform backpropagation
            loss.backward()
            # Update parameters
            optim.step()

            # Training stats
            total += 1
            running_loss += loss.item()

            # Evaluate and track improvements on the validation dataset
            training_accuracy = self.evaluate(datasets['train'])
            validation_accuracy = self.evaluate(datasets['val'])
            if validation_accuracy > best_validation_accuracy and epoch > 30:
                best_validation_accuracy = validation_accuracy
                tmp = self.best_model
                self.best_model = copy.deepcopy(self.state_dict())
                del tmp
            epoch_loss = running_loss / total
            if not epoch % 50:
              print(
                  f"""Epoch: {epoch} Loss: {epoch_loss:.4f} Train acc: {training_accuracy} Val acc: {validation_accuracy}""")
            del loss, preds
            torch.cuda.empty_cache()
            gc.collect()
        
        del optim, weights, criterion
        torch.cuda.empty_cache()
        gc.collect()
        self.load_state_dict(self.best_model)
