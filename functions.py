import os
import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GINConv
from torch_scatter import scatter_mean
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import numpy as np
import copy
import pubchempy as pcp
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to get PubChem fingerprint from SMILES
def PubChemFP(smi):
    pubchem_compound = pcp.get_compounds(smi, 'smiles')[0]
    return [int(bit) for bit in pubchem_compound.cactvs_fingerprint]

# Function to calculate similarity between two PubChem fingerprints
def pubchemsim(pc_fp1, pc_fp2):
    fp1 = set([ind for ind, x in enumerate(pc_fp1) if x == 1])
    fp2 = set([ind for ind, x in enumerate(pc_fp2) if x == 1])
    return float(len(fp1.intersection(fp2))) / float(len(fp1.union(fp2)))

# Function to generate mixture graphs from a dataframe
def generate_mixture_graphs(df, cid_smiles_df, fingerprints=['MACCS']):
    new_rows = []
    mixture_graph_df = pd.DataFrame(columns=['Dataset', 'Mixture', 'ID'])

    for ind, row in df.iterrows():
        cids = [x for x in row[2:].to_list() if x > 0]
        new_row = {'Dataset': row.iloc[0], 'Mixture': row.iloc[1], 'ID': f"{row.iloc[0]}_{row.iloc[1]}"}
        for fingerprint in fingerprints:
            G = nx.Graph()
            for cid in cids:
                try:
                    smi = cid_smiles_df['SMILES'][cid_smiles_df['CID'] == cid].values[0]
                except:
                    print(cid, 'not found')
                if fingerprint == 'MACCS':
                    maccs = cid_smiles_df['maccs_fp'][cid_smiles_df['SMILES'] == smi].item()
                    G.add_node(cid, maccs=maccs)
                    G.add_node(cid, x=torch.tensor(maccs, dtype=torch.float32))
                elif fingerprint == 'ECFP4':
                    ecfp4 = cid_smiles_df['morgan_fp'][cid_smiles_df['SMILES'] == smi].item()
                    G.add_node(cid, ecfp4=ecfp4)
                    G.add_node(cid, x=torch.tensor(list(ecfp4), dtype=torch.float32))
                elif fingerprint == 'pubchem':
                    pubchem_fp = cid_smiles_df['pubchem_fp'][cid_smiles_df['SMILES'] == smi].item()
                    G.add_node(cid, pubchem_fp=pubchem_fp)
                    G.add_node(cid, x=torch.tensor(list(pubchem_fp), dtype=torch.float32))

            for pair in itertools.combinations(cids, 2):
                if fingerprint == 'MACCS':
                    sim = DataStructs.FingerprintSimilarity(G.nodes[pair[0]]['maccs'], G.nodes[pair[1]]['maccs'])
                elif fingerprint == 'ECFP4':
                    sim = DataStructs.TanimotoSimilarity(G.nodes[pair[0]]['ecfp4'], G.nodes[pair[1]]['ecfp4'])
                elif fingerprint == 'pubchem':
                    sim = pubchemsim(G.nodes[pair[0]]['pubchem_fp'], G.nodes[pair[1]]['pubchem_fp'])
                if sim > 0.3:
                    G.add_edge(pair[0], pair[1], weight=sim)
            if len(G.edges) == 0:
                for node in G.nodes:
                    G.add_edge(node, node, weight=1.0)
            new_row['Graph_' + fingerprint] = G
        new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)
    mixture_graph_df = pd.concat([mixture_graph_df, new_rows_df], ignore_index=True)
    return mixture_graph_df

def test_generate_mixture_graphs(df, cid_smiles_df, fingerprints=['MACCS']):
    new_rows = []
    mixture_graph_df = pd.DataFrame(columns=['Dataset', 'Mixture', 'ID'])

    for ind, row in df.iterrows():
        cids = [x for x in row[2:].to_list() if x > 0]
        new_row = {'Dataset': row.iloc[0], 'Mixture': row.iloc[0], 'ID': f"{row.iloc[0]}_{row.iloc[0]}"}
        for fingerprint in fingerprints:
            G = nx.Graph()
            for cid in cids:
                try:
                    smi = cid_smiles_df['SMILES'][cid_smiles_df['CID'] == cid].values[0]
                except:
                    print(cid, 'not found')
                if fingerprint == 'MACCS':
                    maccs = cid_smiles_df['maccs_fp'][cid_smiles_df['SMILES']==smi].item()
                    G.add_node(cid, maccs = maccs)
                    G.add_node(cid, x=torch.tensor(maccs, dtype=torch.float32))
                    
                elif fingerprint == 'ECFP4':
                    ecfp4 = cid_smiles_df['morgan_fp'][cid_smiles_df['SMILES']==smi].item()
                    G.add_node(cid, ecfp4 = ecfp4)
                    G.add_node(cid, x=torch.tensor(list(ecfp4), dtype=torch.float32))

                elif fingerprint == 'pubchem':
                    pubchem_fp = cid_smiles_df['pubchem_fp'][cid_smiles_df['SMILES']==smi].item()
                    G.add_node(cid, pubchem_fp = pubchem_fp)
                    G.add_node(cid, x=torch.tensor(list(pubchem_fp), dtype=torch.float32))
                    
            for pair in itertools.combinations(cids, 2):
                if fingerprint == 'MACCS':
                    sim = DataStructs.FingerprintSimilarity(G.nodes[pair[0]]['maccs'], G.nodes[pair[1]]['maccs'])
                elif fingerprint == 'ECFP4':
                    sim = DataStructs.TanimotoSimilarity(G.nodes[pair[0]]['ecfp4'], G.nodes[pair[1]]['ecfp4'])
                elif fingerprint == 'pubchem':
                    sim = pubchemsim(G.nodes[pair[0]]['pubchem_fp'], G.nodes[pair[1]]['pubchem_fp'])
                if sim > 0.3:
                    G.add_edge(pair[0], pair[1], weight=sim)
            if len(G.edges) == 0:  # If no edges, add self-loops
                for node in G.nodes:
                    G.add_edge(node, node, weight=1.0)
            new_row['Graph_'+fingerprint] = G
        new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)
    mixture_graph_df = pd.concat([mixture_graph_df, new_rows_df], ignore_index=True)
    return mixture_graph_df

# Merge training graphs with similarity data
def training_mixture_merged_df(graph_df, sim_df):
    merged_df = pd.merge(graph_df, sim_df, right_on=['Dataset', 'Mixture 1'], left_on=['Dataset', 'Mixture'])
    merged_df.drop(['Mixture'], axis=1, inplace=True)
    merged_df.rename(columns={'ID': 'ID_1', 'Graph_MACCS': 'Graph_1_MACCS', 'Graph_ECFP4': 'Graph_1_ECFP4', 'Graph_pubchem': 'Graph_1_pubchem'}, inplace=True)

    merged_df = pd.merge(graph_df, merged_df, right_on=['Dataset', 'Mixture 2'], left_on=['Dataset', 'Mixture'])
    merged_df.drop(['Mixture'], axis=1, inplace=True)
    merged_df.rename(columns={'ID': 'ID_2', 'Graph_MACCS': 'Graph_2_MACCS', 'Graph_ECFP4': 'Graph_2_ECFP4', 'Graph_pubchem': 'Graph_2_pubchem'}, inplace=True)
    merged_df.rename(columns={'Experimental Values': 'Distance'}, inplace=True)

    merged_df['Graph_union_MACCS'] = merged_df.apply(lambda x: nx.compose(x['Graph_1_MACCS'], x['Graph_2_MACCS']), axis=1)
    merged_df['Graph_union_ECFP4'] = merged_df.apply(lambda x: nx.compose(x['Graph_1_ECFP4'], x['Graph_2_ECFP4']), axis=1)
    merged_df['Graph_union_pubchem'] = merged_df.apply(lambda x: nx.compose(x['Graph_1_pubchem'], x['Graph_2_pubchem']), axis=1)
    return merged_df

# merge the test similarity values with the test graph dataframe
def test_mixture_merged_df(graph_df, sim_df):
    # mixture 1
    mergend_df = pd.merge(graph_df, sim_df, right_on=['Mixture_1'], left_on=['Mixture'])
    mergend_df.drop(['Mixture'], axis=1, inplace=True)
    mergend_df.drop(['Dataset_y'], axis=1, inplace=True)
    mergend_df.rename(columns={'ID':'ID_1', 'Graph_MACCS':'Graph_1_MACCS', 'Graph_ECFP4':'Graph_1_ECFP4', 'Graph_pubchem':'Graph_1_pubchem'}, inplace=True)

    # mixture 2
    mergend_df = pd.merge(graph_df, mergend_df, right_on=['Mixture_2'], left_on=['Mixture'])
    mergend_df.drop(['Dataset_x'], axis=1, inplace=True)
    mergend_df.rename(columns={'ID':'ID_2', 'Graph_MACCS':'Graph_2_MACCS', 'Graph_ECFP4':'Graph_2_ECFP4', 'Graph_pubchem':'Graph_2_pubchem'}, inplace=True)
    mergend_df.rename(columns={'Predicted_Experimental_Values':'Distance'}, inplace=True)

    # union graph
    mergend_df['Graph_union_MACCS'] = mergend_df.apply(lambda x: nx.compose(x['Graph_1_MACCS'], x['Graph_2_MACCS']), axis=1)
    mergend_df['Graph_union_ECFP4'] = mergend_df.apply(lambda x: nx.compose(x['Graph_1_ECFP4'], x['Graph_2_ECFP4']), axis=1)
    mergend_df['Graph_union_pubchem'] = mergend_df.apply(lambda x: nx.compose(x['Graph_1_pubchem'], x['Graph_2_pubchem']), axis=1)
    return mergend_df

# Custom Dataset Class
class GraphPairDataset(Dataset):
    def __init__(self, graph_pairs, labels, transform=None, pre_transform=None):
        self.graph_pairs = graph_pairs
        self.labels = labels
        super(GraphPairDataset, self).__init__('.', transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.graph_pairs))]

    def process(self):
        for i, (pair, label) in enumerate(zip(self.graph_pairs, self.labels)):
            data_list = []
            for fp in pair:
                for G in fp:
                    data = from_networkx(G)
                    if G.number_of_edges() > 0:
                        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                        data.edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
                    data_list.append(data)
            torch.save((data_list, label), self.processed_paths[i])

    def len(self):
        return len(self.graph_pairs)

    def get(self, idx):
        data_list, label = torch.load(self.processed_paths[idx])
        return data_list, label

# Generate pairs of graphs with labels
def generate_graph_pairs(_df, fingerprints):
    pairs, labels = [], []
    for ind, row in _df.iterrows():
        graph_pairs = []
        for fingerprint in fingerprints:
            graph_pair = row[['Graph_1_' + fingerprint, 'Graph_2_' + fingerprint, 'Graph_union_' + fingerprint]].to_list()
            graph_pairs.append(graph_pair)
        pairs.append(graph_pairs)
        labels.append(torch.Tensor([row['Distance']]).float())
    return pairs, labels

# Data loaders
def collate_fn(batch):
    data_lists = [[] for _ in range(len(batch[0][0]))]  # Create empty lists for each fingerprint
    labels = []
    for data, label in batch:
        for i, graph in enumerate(data):
            data_lists[i].append(graph)
        labels.append(label)
    return data_lists, torch.tensor(labels)

def LinearBlockWithSigmoid(input_dim, output_dim):  # similarity(fc1)
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.Sigmoid()
    )

def LinearBlock(input_dim, output_dim): # comparison (fc2)
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU()
    )

class SimpleModel(nn.Module):
    def __init__(self, fp_size=16):
        super(SimpleModel, self).__init__()
        self.fp_size = fp_size
        nn1 = nn.Sequential(nn.Linear(self.fp_size, 64), nn.ReLU(), nn.Linear(64, 64))
        nn2 = nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Linear(8, 8))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(8)
        self.dropout_rate = 0.2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = scatter_mean(x, batch, dim=0)
        return x.unsqueeze(1)

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings):
        attention_scores = self.attention_fc(embeddings)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_embeddings = embeddings * attention_weights
        aggregated_embeddings = torch.sum(weighted_embeddings, dim=1)
        return aggregated_embeddings, weighted_embeddings

class GraphSimilarityModel(nn.Module):
    def __init__(self):
        super(GraphSimilarityModel, self).__init__()
        self.model_maccs = SimpleModel(fp_size=167)
        self.model_ecfp = SimpleModel(fp_size=1024)
        self.model_pubchem = SimpleModel(fp_size=881)
        self.fc1 = LinearBlockWithSigmoid(16, 1)  # Linear layer to output similarity value (32 + 32)
        self.fc2 = LinearBlock(16, 8)  # Linear layer to compare with graph_union
        self.attention = AttentionLayer(8)

    def prepare_data(self, graph):
        graph.x = graph.x.float()  # Convert node features to float
        if graph.edge_index.dtype != torch.long:
            graph.edge_index = graph.edge_index.long()  # Convert edge indices to long, if not already
        return graph
    
    def forward(self, data):
        g1_maccs, g2_maccs, g_u_maccs, g1_ecfp, g2_ecfp, g_u_ecfp, g1_pubchem, g2_pubchem, g_u_pubchem = data  # Unpack the three graphs for each of the fingerprints
        g1_maccs, g2_maccs, g_u_maccs = self.prepare_data(g1_maccs), self.prepare_data(g2_maccs), self.prepare_data(g_u_maccs)
        g1_ecfp, g2_ecfp, g_u_ecfp = self.prepare_data(g1_ecfp), self.prepare_data(g2_ecfp), self.prepare_data(g_u_ecfp)
        g1_pubchem, g2_pubchem, g_u_pubchem = self.prepare_data(g1_pubchem), self.prepare_data(g2_pubchem), self.prepare_data(g_u_pubchem)
        emb1_maccs = self.model_maccs(g1_maccs)  
        emb2_maccs = self.model_maccs(g2_maccs)  
        emb3_maccs = self.model_maccs(g_u_maccs)  
        emb1_ecfp = self.model_ecfp(g1_ecfp)  
        emb2_ecfp = self.model_ecfp(g2_ecfp)  
        emb3_ecfp = self.model_ecfp(g_u_ecfp)  
        emb1_pubchem = self.model_pubchem(g1_pubchem)  
        emb2_pubchem = self.model_pubchem(g2_pubchem)  
        emb3_pubchem = self.model_pubchem(g_u_pubchem)  

        emb_1_agg = torch.cat((emb1_maccs, emb1_ecfp, emb1_pubchem), dim=1)  
        emb_1_agg, emb_1_att_w = self.attention(emb_1_agg)
        emb_2_agg = torch.cat((emb1_maccs, emb1_ecfp, emb1_pubchem), dim=1)  
        emb_2_agg, emb_2_att_w = self.attention(emb_2_agg)
        emb_u_agg = torch.cat((emb1_maccs, emb1_ecfp, emb1_pubchem), dim=1)  
        emb_u_agg, emb_u_att_w = self.attention(emb_u_agg)

        combined_emb = torch.cat((emb_1_agg, emb_2_agg), dim=1)
        similarity = self.fc1(combined_emb)
        comparison = self.fc2(combined_emb)

        return similarity, comparison, emb_u_agg
    
def mse_loss(pred, target):
    target = target.view(-1)
    return F.mse_loss(pred, target)

def train(train_loader, model, optimizer, device):
    model.train()
    total_loss, total_loss_sim, total_loss_emb = 0, 0, 0
    for data_list, y in train_loader:
        optimizer.zero_grad()
        data_list = [d.to(device) for d in data_list]
        y = y.float().to(device)
        out, emb12, emb3 = model(data_list)
        predicted_distance = 1 - out.view(-1)
        loss_sim = mse_loss(predicted_distance, y)
        emb_distance = 1 - F.pairwise_distance(emb12, emb3, p=2)
        loss_emb = mse_loss(emb_distance, y)
        loss = loss_sim + loss_emb
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss_sim += loss_sim.item()
        total_loss_emb += loss_emb.item()
    return total_loss / len(train_loader), total_loss_sim / len(train_loader), total_loss_emb / len(train_loader)

def evaluate(loader, model, device):
    model.eval()
    total_loss, total_loss_sim, total_loss_emb = 0, 0, 0
    preds, targets = [], []

    with torch.no_grad():
        for data_list, y in loader:
            data_list = [d.to(device) for d in data_list]
            y = y.float().to(device)
            out, emb12, emb3 = model(data_list)
            predicted_distance = 1 - out.view(-1)
            loss_sim = mse_loss(predicted_distance, y.view(-1))
            emb_distance = 1 - F.pairwise_distance(emb12, emb3, p=2)
            loss_emb = mse_loss(emb_distance, y)
            loss = loss_sim + loss_emb
            total_loss += loss.item()
            total_loss_sim += loss_sim.item()
            total_loss_emb += loss_emb.item()
            preds.extend(predicted_distance.cpu().numpy())
            targets.extend(y.view(-1).cpu().numpy())
    pearson_corr, _ = pearsonr(preds, targets)
    return total_loss / len(loader), total_loss_sim / len(loader), total_loss_emb / len(loader), pearson_corr

def format_to_four_decimals(data):
    return [f"{x:.4f}" if isinstance(x, (int, float)) else x for x in data]

def save_csv(data, filename, header):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

def save_best_metrics(best_metrics, filename):
    header = [
        'epoch', 'train_loss', 'train_loss_sim', 'train_loss_emb', 'val_loss', 'val_loss_sim', 'val_loss_emb',
        'val_pearson_corr', 'test_loss', 'test_loss_sim', 'test_loss_emb', 'test_pearson_corr'
    ]
    data = [[
        best_metrics['epoch'], best_metrics['train_loss'], best_metrics['train_loss_sim'], best_metrics['train_loss_emb'],
        best_metrics['val_loss'], best_metrics['val_loss_sim'], best_metrics['val_loss_emb'], best_metrics['val_pearson_corr'],
        best_metrics['test_loss'], best_metrics['test_loss_sim'], best_metrics['test_loss_emb'], best_metrics['test_pearson_corr'],
    ]]
    formatted_data = [format_to_four_decimals(row) for row in data]
    if not os.path.exists(filename):
        save_csv(data, filename, header)
    else:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(formatted_data)

def load_best_model(run_number, model_class, model_dir='best_models'):
    model_path = os.path.join(model_dir, f'train.py_best_model_run_{run_number}.pth')
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data_list, _ in loader:
            data_list = [d.to(device) for d in data_list]
            out, _, _ = model(data_list)
            predicted_distance = 1 - out.view(-1)
            predictions.extend(predicted_distance.cpu().numpy())
    return np.array(predictions)
