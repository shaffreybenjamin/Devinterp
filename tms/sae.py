# %%
import os
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
from scipy import stats
from typing import Tuple
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import copy
from scipy.optimize import linear_sum_assignment

from toy_model import train_toy_model, load_model, get_dataloader, ToyModel

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, l1_lambda: float = 0.001):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.l1_lambda = l1_lambda

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        activations = self.encoder(x)
        reconstructed = self.decoder(F.relu(activations))
        return activations, reconstructed
        
    def loss(self, x: torch.Tensor, norm: torch.float = 0.0) -> torch.Tensor:
        activations, reconstructed = self(x)
        reconstruction_loss = F.mse_loss(reconstructed, x)
        sparsity_loss = self.l1_lambda * activations.abs().mean()
        #print(f'loss terms: rec_err {reconstruction_loss},  sparsity {sparsity_loss}, frob_norm {0.001 * norm}')
        return reconstruction_loss + sparsity_loss + .00001 * norm

class DisentangledModel(nn.Module):
    def __init__(self, toy_model: ToyModel, sae: SparseAutoencoder):
        super().__init__()
        self.n_features = toy_model.W.shape[1]  # c
        self.n_hidden = toy_model.W.shape[0]    # r
        self.n_dict = sae.encoder.weight.shape[0]  # d
        self.importances = toy_model.importances
        
        # Create encoder and decoder as Sequential modules
        self.encoder = nn.Sequential(
            # First apply toy model projection: c -> r
            nn.Linear(self.n_features, self.n_hidden, bias=False),
            # Then apply SAE encoder: r -> d 
            nn.Linear(self.n_hidden, self.n_dict, bias=False),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            # First apply SAE decoder: d -> r
            nn.Linear(self.n_dict, self.n_hidden, bias=False),
            # Then project back using toy model: r -> c
            nn.Linear(self.n_hidden, self.n_features, bias=True),
            nn.ReLU()
        )
        
        # Initialize weights from provided models
        with torch.no_grad():
            # Encoder weights
            self.encoder[0].weight.data = toy_model.W.data
            self.encoder[1].weight.data = sae.encoder.weight.data
            
            # Decoder weights  
            self.decoder[0].weight.data = sae.decoder.weight.data
            self.decoder[1].weight.data = toy_model.W.data.T
            self.decoder[1].bias.data = toy_model.b.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying encoder then decoder"""
        h = self.encoder(x)
        return self.decoder(h)

    def get_hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get activations from encoder, matching toy model API"""
        return self.encoder(x)


class SAEEvaluator:
    def __init__(self, sae_model: SparseAutoencoder, toy_model, device=None):
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sae_model = sae_model.to(device)
        self.toy_model = toy_model.to(device)
        self.device = device
        toy_weights = self.toy_model.W.data.T
        sae_weights = self.sae_model.encoder.weight.data
        self.current_permutation = torch.argmax(self.compute_similarities(toy_weights, sae_weights), dim=1)

    def compute_similarities(self, toy_weights, sae_weights):
        return F.cosine_similarity(
            sae_weights.unsqueeze(0),
            toy_weights.unsqueeze(1),
            dim=2
        )

    def get_best_permutation(self, toy_weights, sae_weights):
        """Return indices that would reorder SAE weights to best match toy weights"""
        similarities = self.compute_similarities(toy_weights, sae_weights)
        return torch.argmax(similarities, dim=1)

    def feature_learning_score(self, toy_weights, sae_weights):
        similarities = self.compute_similarities(toy_weights, sae_weights)
        max_similarities = similarities.max(dim=1)[0]
        return max_similarities.mean()

    def weighted_feature_learning_score(self, toy_weights, sae_weights, importances):
        similarities = self.compute_similarities(toy_weights, sae_weights)
        max_similarities = similarities.max(dim=1)[0]
        return (max_similarities * importances).sum() / importances.sum()

    def reconstruction_score(self, test_dataloader):
        
        for batch in test_dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                hidden = self.toy_model.get_hidden_activations(batch)
                _, sae_reconstruction = self.sae_model(hidden)
                mse = F.mse_loss(sae_reconstruction, hidden)
                
        return torch.exp(-mse)

    def permutation_dist(self, toy_weights, sae_weights):
        new_permutation = self.get_best_permutation(toy_weights, sae_weights)
        perm_dist = torch.abs(self.current_permutation - new_permutation).sum()
        self.current_permutation = new_permutation
        return perm_dist

    def compute_sparsity(self, test_dataloader):
        
        for batch in test_dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                hidden = self.toy_model.get_hidden_activations(batch)
                sae_activations, _ = self.sae_model(hidden)
                toy_sparsity = (hidden != 0).float().mean()
                sae_sparsity = (sae_activations != 0).float().mean()
        return abs(toy_sparsity - sae_sparsity)

    def reorder_sae_weights(self):
        """Reorder both encoder and decoder weights according to best permutation"""
        with torch.no_grad():
            # Reorder encoder weights
            n_features = self.toy_model.W.data.T.shape[0]
        
            # Get best permutation for first n_features columns
            perm = self.current_permutation
        
            # Create full permutation that keeps remaining columns in place
            full_perm = torch.arange(self.sae_model.encoder.weight.data.shape[0], device=self.device)
            full_perm[:n_features] = perm[:n_features]
            
            self.sae_model.encoder.weight.data = self.sae_model.encoder.weight.data[full_perm]
            
            # Reorder decoder weights (in opposite direction)
            self.sae_model.decoder.weight.data = self.sae_model.decoder.weight.data[:, full_perm]

    def get_disentangled_model(self):
        """Return disentangled model weights after reordering"""
        with torch.no_grad():
            n_features = self.toy_model.W.data.T.shape[0]
        
            '''
            # Get best permutation for first n_features columns
            perm = self.current_permutation
        
            # Create full permutation that keeps remaining columns in place
            full_perm = torch.arange(self.sae_model.encoder.weight.data.shape[0], device=self.device)
            full_perm[:n_features] = perm[:n_features]
            '''

            disentangled_model = DisentangledModel(self.toy_model, self.sae_model)
            # self.sae_model.encoder.weight.data @ disentangled_model.W.data
            # d x r, r x c W.T W = c x c, d x c, W.T W = c x c 
            #disentangled_model.encoder.W.data = self.sae_model.decoder.weight.data @ self.sae_model.encoder.weight.data @ disentangled_model.W.data + disentangled_model.b.data
            #disentangled_model.encoder.W.data = self.sae_model.encoder.weight.data @ self.toy_model.W.data
            # include relu is using decoder of SAE
            #disentangled_model.decoder.W.data = self.sae_model.decoder.weight.data @ self.toy_model.W.data.T
            # pseudoinverse of encoder times transpose of toy matrix W

            
            
            return disentangled_model

    def align_features(self, toy_weights, sae_weights):
        """
        Align SAE features with ground truth features using Hungarian algorithm.
    
        Args:
            toy_weights: Ground truth feature matrix (n_features × n_hidden)
            sae_weights: SAE decoder weights (n_features × n_hidden)
    
        Returns:
            optimal_perm: Optimal permutation of SAE features
            similarity_score: Overall matching quality
        """
        # Compute pairwise cosine similarity matrix
        similarities = self.compute_similarities(toy_weights, sae_weights)
    
        # Convert to cost matrix (Hungarian minimizes cost)
        cost_matrix = 1 - similarities
    
        # Find optimal assignment
        true_idx, sae_idx = linear_sum_assignment(cost_matrix.cpu())
    
        # Compute overall similarity score
        similarity_score = similarities[true_idx, sae_idx].mean()
    
        return sae_idx, similarity_score

    def evaluate(self, test_dataloader):
        toy_features = self.toy_model.W.data.T
        sae_features = self.sae_model.encoder.weight.data
        #sae_features = select_top_k(sae_features, toy_features.shape[1])

        results = {
            'mean_max_cs': self.feature_learning_score(toy_features, sae_features),
            'weighted_mean_max_cs': self.weighted_feature_learning_score(
                toy_features, sae_features, self.toy_model.importances),
            'reconstruction_mse_score': self.reconstruction_score(test_dataloader),
            'permutation_dist': self.permutation_dist(toy_features, sae_features),
            'sparsity_difference': self.compute_sparsity(test_dataloader),
            'disentangled_model': self.get_disentangled_model()
        }
        return results

def train_sparse_autoencoder(toy_model, evaluator, sparsity, n_dictionary, l1_lambda, sae = None, optimizer = None, n_epochs=1, dataloader = None, disable_tqdm=False):
    """Train a sparse autoencoder on toy model hidden activations."""
    n_features = toy_model.W.shape[1]
    n_hidden = toy_model.W.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initial hyperparameters
    l1_lambda_start = 0.001  # Start with a small L1 penalty
    lr_start = l1_lambda
    lr_final = 0.0001
    
    if dataloader == None:
        dataloader = get_dataloader(sparsity, n_features)

    if sae == None:
        sae = SparseAutoencoder(n_hidden, n_dictionary, l1_lambda=l1_lambda)
        optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)

    # Move model to device
    sae = sae.to(device)
    
    for epoch in tqdm(range(n_epochs), disable=disable_tqdm):
        # Calculate current l1_lambda and learning rate
        progress = epoch / (n_epochs - 1)  # Goes from 0 to 1
        current_l1 = l1_lambda_start + (l1_lambda - l1_lambda_start) * progress
        current_lr = lr_start * (lr_final / lr_start) ** progress
        
        # Update hyperparameters
        sae.l1_lambda = current_l1
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        for batch in dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                hidden = toy_model.get_hidden_activations(batch)
            optimizer.zero_grad()
            toy_features = toy_model.W.data.T
            sae_features = sae.encoder.weight.data
            sae_features = select_top_k(sae_features, n_features)
    
            WTW_sae = torch.matmul(sae_features, sae_features.T)
            WTW_toy = torch.matmul(toy_features, toy_features.T)
            frob_norm = torch.norm(WTW_toy - WTW_sae, p='fro')
            #loss = sae.loss(hidden, frob_norm)
            loss = sae.loss(hidden)
            loss.backward()
            optimizer.step()
            
        toy_features = toy_model.W.data.T
        sae_features = sae.encoder.weight.data
        sae_features = select_top_k(sae_features, n_features)
        sae_idx, similarity_score = evaluator.align_features(toy_features, sae_features)
    
        '''
        WTW_sae = torch.matmul(sae_features, sae_features.T)
        WTW_toy = torch.matmul(toy_features, toy_features.T)
        frob_norm = torch.norm(WTW_toy - WTW_sae, p='fro')
        
        if frob_norm <= 1:
            print(f'achieved good fit for SAE with Frob norm of diff {frob_norm}')
            return sae, optimizer
        '''
        
        '''
        if similarity_score >= 0.9:
            print(f'achieved good fit for SAE with {similarity_score}')
            return sae, optimizer
        '''
        
            
    return sae, optimizer

def visualize_sae_and_toy_model(W_sae, W_toy, sparsity, epoch='', save_path='results/'):
    """Visualize the WᵀW matrices of SAE and toy model side by side."""
    WTW_sae = torch.matmul(W_sae, W_sae.T)
    WTW_toy = torch.matmul(W_toy, W_toy.T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(WTW_sae.cpu().numpy(), ax=ax1, cmap='coolwarm', center=0, square=True,
                cbar=True, xticklabels=False, yticklabels=False)
    ax1.set_title("SAE WᵀW")

    sns.heatmap(WTW_toy.cpu().numpy(), ax=ax2, cmap='coolwarm', center=0, square=True,
                cbar=True, xticklabels=False, yticklabels=False)
    ax2.set_title("Toy Model WᵀW")

    plt.suptitle(f"Sparsity: {sparsity}")
    plt.tight_layout()
    plt.savefig(f'{save_path}/sae_{sparsity}_{epoch}.png')
    #plt.show()
    plt.close()
    # Ensure memory is cleared
    fig.clf()
    plt.close('all')

def get_sparsities(number_of_sparsities):
    sparsities = np.logspace(-3, 0, number_of_sparsities)
    sparsities = 1 - sparsities[::-1]  # Convert to sparsities from 0 to 0.999
    sparsities = [np.round(sparsity, 5).item() for sparsity in sparsities]
    sparsities = [0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.97, 0.99, 0.997, 0.999]
    return [0.7]

def select_top_k(W, k):
    diag = torch.diag(torch.matmul(W, W.T))
    diag, indices = torch.topk(diag, k=k)
    return W[indices]

def train_sweep_of_sae_on_toy_model(n_hidden, n_features, n_dictionary, l1_lambda, save_path, n_sparsities=10, epochs=100, log_freq=1, n_samples=50000, batch_size=512, limit=False, temperature=1.0, criterion=None):
    """Train a sweep of SAEs on toy models with different sparsities."""
    sparsities = get_sparsities(n_sparsities)
    for sparsity in tqdm(sparsities):

        train_losses, test_losses, toy_model, sae, toy_models_saved, sae_models_saved, disentangled_models_saved, train_loader, mmcs_values, wmmcs_values, similarity_scores, frob_norms, permutation_distances = train_toy_model(sparsity, n_features, n_hidden, n_dictionary, epochs=epochs, log_freq=log_freq, n_samples=n_samples, batch_size=batch_size, limit=limit, temperature=temperature, criterion=criterion)
        #evaluator = SAEEvaluator(sae, toy_model)
        #sae_after, optimizer = train_sparse_autoencoder(toy_model, evaluator, sparsity, n_dictionary, l1_lambda, None, None, n_epochs=100, dataloader=train_loader)
        sae_after = None

        torch.save(toy_model.state_dict(), f"{save_path}toy_model_s{sparsity}_f{n_features}_h{n_hidden}.pth")

        torch.save(sae.state_dict(), f"{save_path}sae_s{sparsity}_d{n_dictionary}_h{n_hidden}.pth")

        W_sae = sae.encoder.weight.data
        W_sae = select_top_k(W_sae, n_features)
        W_toy = toy_model.W.data.T

        visualize_sae_and_toy_model(W_sae, W_toy, sparsity, epochs, save_path)

        '''
        plt.title("SAE W")
        plt.imshow(W_sae.cpu().numpy(), cmap='coolwarm')
        plt.colorbar()
        plt.show()
        '''

    return train_losses, test_losses, toy_models_saved, sae_models_saved, disentangled_models_saved, sae_after, train_loader, mmcs_values, wmmcs_values, similarity_scores, frob_norms, permutation_distances




'''
# %%
if __name__ == "__main__":

    n_hidden = 5
    n_features = 20
    n_dictionary = 40
    l1_lambda = 0.01
    n_sparsities = 10
    save_path = f"~/results/"
    os.makedirs(save_path, exist_ok=True)
    train_sweep_of_sae_on_toy_model(n_hidden, n_features, n_dictionary, l1_lambda, save_path, n_sparsities=n_sparsities)

# %%
    sparsities = get_sparsities(n_sparsities)
    for sparsity in sparsities: 
        sae_state_dict = torch.load(f"{save_path}sae_s{sparsity}_d{n_dictionary}_h{n_hidden}.pth")
        
        sae = SparseAutoencoder(n_hidden, n_dictionary)
        sae.load_state_dict(sae_state_dict)
        W_sae = sae.encoder.weight.data

        toy_model = load_model(sparsity, n_features, n_hidden)
        W_toy = toy_model.W.data.T


        def select_top_k(W, k):
            diag = torch.diag(torch.matmul(W, W.T))
            diag, indices = torch.topk(diag, k=k)
            return W[indices]
       
        W_sae = select_top_k(W_sae, n_features)
        visualize_sae_and_toy_model(W_sae, W_toy, sparsity)

        plt.title("SAE W")
        plt.imshow(W_sae.numpy(), cmap='coolwarm')
        plt.colorbar()
        plt.show()

'''