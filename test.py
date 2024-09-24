import os
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from functions import *

# Set device to CUDA if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_number = 2

# Load data
cid_smiles_df = pd.read_pickle('Dream/data/cid_smiles_df.pkl')
mixture_test_df = pd.read_csv('Dream/data/Mixure_Definitions_test_set.csv')
mixture_test_sim_df = pd.read_csv('Dream/data/Test_set.csv')

# Fingerprints and hyperparameters
fingerprints = ['MACCS', 'ECFP4', 'pubchem']

# Generate graphs for test datasets
mixture_test_graph_df = test_generate_mixture_graphs(mixture_test_df, cid_smiles_df, fingerprints=fingerprints)

# Merge graphs with similarity data
mixture_test_merged_df = test_mixture_merged_df(mixture_test_graph_df, mixture_test_sim_df)

# Generate graph pairs and labels for each dataset
final_test_pairs, final_test_labels = generate_graph_pairs(mixture_test_merged_df, fingerprints)

# Create graph pair datasets
final_test_dataset = GraphPairDataset(final_test_pairs, final_test_labels)

# Ensure datasets are processed
final_test_dataset.process()

# Data loader for the test dataset
final_test_loader = DataLoader(final_test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# Load the best model from the training
best_model = load_best_model(run_number, model_class=GraphSimilarityModel, model_dir='best_models')

# Make predictions on the test dataset
final_test_predictions = predict(best_model, final_test_loader, device)

# Save the predictions to a CSV file
mixture_test_sim_df['Predicted_Experimental_Values'] = final_test_predictions
mixture_test_sim_df.to_csv('final_test_predictions.csv', index=False)