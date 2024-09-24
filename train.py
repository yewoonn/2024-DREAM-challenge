import os
import torch
from torch_geometric.loader import DataLoader
from torch.optim import lr_scheduler
import copy
import pandas as pd
from functions import *

# Enable CUDA launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read Pickle and CID csv
cid_smiles_df = pd.read_pickle('Dream/data/cid_smiles_df.pkl')
mixture_training_df = pd.read_csv('Dream/data/Mixure_Definitions_Training_set.csv')
mixture_training_sim_df = pd.read_csv('Dream/data/TrainingData_mixturedist.csv')

# Fingerprints and hyperparameters
fingerprints = ['MACCS', 'ECFP4', 'pubchem']
train_size, val_size, test_size = 0.7, 0.15, 0.15

# Create folder for best models
best_models_folder = 'best_models'
os.makedirs(best_models_folder, exist_ok=True)

folder_name = 'Result'
os.makedirs(folder_name, exist_ok=True)

current_file_name = os.path.basename(__file__)

edge_cutoff_best_metrics = []

for run in range(10):
    print(f"File name: {current_file_name} Run:{run + 1} \n")

    model = GraphSimilarityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    mixture_training_graph_df = generate_mixture_graphs(mixture_training_df, cid_smiles_df, fingerprints=fingerprints)
    mixture_training_merged_df = training_mixture_merged_df(mixture_training_graph_df, mixture_training_sim_df)

    training_train_df, temp_df = train_test_split(mixture_training_merged_df, test_size=(1 - train_size), random_state=42)
    training_val_df, training_test_df = train_test_split(temp_df, test_size=(test_size / (val_size + test_size)), random_state=42)

    training_train_pairs, training_train_labels = generate_graph_pairs(training_train_df, fingerprints)
    training_val_pairs, training_val_labels = generate_graph_pairs(training_val_df, fingerprints)
    training_test_pairs, training_test_labels = generate_graph_pairs(training_test_df, fingerprints)

    training_train_dataset = GraphPairDataset(training_train_pairs, training_train_labels)
    training_val_dataset = GraphPairDataset(training_val_pairs, training_val_labels)
    training_test_dataset = GraphPairDataset(training_test_pairs, training_test_labels)

    training_train_loader = DataLoader(training_train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    training_val_loader = DataLoader(training_val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    training_test_loader = DataLoader(training_test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    num_epochs = 500
    min_epochs = 250
    patience = 10
    min_delta = 1e-3
    patience_counter = 0

    previous_val_loss = np.inf
    previous_test_loss = np.inf
    lr_list = []
    best_val_loss = float('inf')
    best_model_state = None
    best_metrics = {}
    stable_counter = 0
    epoch_data = []

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        train_loss, train_loss_sim, train_loss_emb = train(training_train_loader, model, optimizer, device)
        val_loss, val_loss_sim, val_loss_emb, val_pearson_corr = evaluate(training_val_loader, model, device)
        test_loss, test_loss_sim, test_loss_emb, test_pearson_corr = evaluate(training_test_loader, model, device)

        epoch_data.append(format_to_four_decimals([
            epoch, current_lr, train_loss, train_loss_sim, train_loss_emb,
            val_loss, val_loss_sim, val_loss_emb, val_pearson_corr,
            test_loss, test_loss_sim, test_loss_emb, test_pearson_corr,
        ]))

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                'epoch': epoch,
                'train_loss': train_loss, 'train_loss_sim': train_loss_sim, 'train_loss_emb': train_loss_emb,
                'val_loss': val_loss, 'val_loss_sim': val_loss_sim, 'val_loss_emb': val_loss_emb, 'val_pearson_corr': val_pearson_corr,
                'test_loss': test_loss, 'test_loss_sim': test_loss_sim, 'test_loss_emb': test_loss_emb, 'test_pearson_corr': test_pearson_corr,
            }
            patience_counter = 0
            stable_counter = 0
        else:
            patience_counter += 1

        if epoch >= min_epochs and np.abs(previous_val_loss - val_loss) < min_delta and np.abs(previous_test_loss - test_loss) < min_delta:
            stable_counter += 1
        else:
            stable_counter = 0

        if epoch >= min_epochs and stable_counter >= patience:
            break

        previous_val_loss = val_loss
        previous_test_loss = test_loss

        scheduler.step()

    csv_filename = os.path.join(folder_name, f'{current_file_name}_results_run_{run + 1}.csv')
    header = ['epoch', 'lr', 'train_loss', 'train_loss_sim', 'train_loss_emb', 'val_loss', 'val_loss_sim', 'val_loss_emb', 'val_pearson_corr',
              'test_loss', 'test_loss_sim', 'test_loss_emb', 'test_pearson_corr']
    save_csv(epoch_data, csv_filename, header)

    if best_metrics:
        edge_cutoff_best_metrics.append(best_metrics)

        model_filename = os.path.join(best_models_folder, f'{current_file_name}_best_model_run_{run + 1}.pth')
        torch.save(best_model_state, model_filename)

edge_cutoff_metrics_filename = os.path.join(folder_name, f'{current_file_name}_best_metrics.csv')
for metrics in edge_cutoff_best_metrics:
    save_best_metrics(metrics, edge_cutoff_metrics_filename)