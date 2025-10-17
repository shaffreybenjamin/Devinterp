# %%
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import copy
import os

class SparseRandomVectorDataset(Dataset):
    def __init__(self, n_samples, n_features, sparsity, limit=False, temperature=1.0):
        self.data = self.generate_data(n_samples, n_features, sparsity, limit, temperature)
    
    def generate_data(self, n_samples, n_features, sparsity, limit=False, temperature=1.0):
        '''
        else:
            samples = []
            while len(samples) < n_samples:
                # Generate a batch of samples (making it larger than needed to be efficient)
                batch_size = max(n_samples - len(samples), n_samples)  # Generate at least n_samples each time
                x = np.random.uniform(0, 1, (batch_size, n_features))
                mask = np.random.rand(*x.shape) < sparsity
                x[mask] = 0
        
                # Keep only valid rows (those with at least one nonzero element)
                valid_rows = x[(x != 0).any(axis=1)]
        
                # Add to our collection
                samples.extend(valid_rows[:n_samples - len(samples)])
    
            # Convert to numpy array and ensure exactly n_samples
            samples = np.array(samples[:n_samples])
            print(f'dataset size is {len(x)}')
            print(f'num samples with more than one nonzero elem is {np.sum(np.count_nonzero(x != 0, axis=1) > 1)}')
        '''
        if limit:
            # Initialize zero tensor to hold samples
            x = np.zeros((n_samples, n_features))
            base = 2
    
            # Calculate the probabilities for each feature according to power law
            # The formula below ensures decreasing probabilities by feature index
            unnormalized_probs = np.array([1/(base ** (i * temperature)) for i in range(n_features)])
            probs = unnormalized_probs / unnormalized_probs.sum()

            print(f'probs: {probs}')
    
            # Sample coordinates according to the power law distribution
            coords = np.random.choice(n_features, size=n_samples, p=probs)
    
            # Sample lengths uniformly
            lengths = np.random.uniform(0, 1, size=n_samples)
    
            # Put lengths in the right coordinates
            x[np.arange(n_samples), coords] = lengths
            for feature in range(n_features):
                print(f'feature {feature}: {len(x[x[ : , feature] != 0])}')
        else:
            samples = []
            base = 2  # Same base as in the limit case
    
            # Calculate feature probabilities according to power law
            unnormalized_probs = np.array([1/(base ** (i * temperature)) for i in range(n_features)])
            feature_probs = unnormalized_probs / unnormalized_probs.sum()
    
            # Calculate expected number of non-zero features per sample based on sparsity
            expected_nonzeros = int(round(n_features * (1 - sparsity)))
            expected_nonzeros = max(1, expected_nonzeros)  # Ensure at least 1 non-zero feature
    
            while len(samples) < n_samples:
                # Generate a batch of samples
                batch_size = max(n_samples - len(samples), n_samples)
                x = np.zeros((batch_size, n_features))
        
                for i in range(batch_size):
                    # Determine how many features will be non-zero for this sample
                    num_nonzeros = np.random.binomial(n_features, 1 - sparsity)
                    num_nonzeros = max(1, num_nonzeros)  # Ensure at least 1 non-zero feature
            
                    # Sample which features will be non-zero according to power law
                    nonzero_features = np.random.choice(
                        n_features, 
                        size=num_nonzeros, 
                        p=feature_probs,
                        replace=False  # Prevent the same feature from being selected twice
                    )
            
                    # Assign uniform random values to the selected features
                    x[i, nonzero_features] = np.random.uniform(0, 1, size=num_nonzeros)
        
                # Keep only valid rows (those with at least one nonzero element)
                valid_rows = x[(x != 0).any(axis=1)]
        
                # Add to our collection
                samples.extend(valid_rows[:n_samples - len(samples)])
    
            # Convert to numpy array and ensure exactly n_samples
            samples = np.array(samples[:n_samples])
            print(f'dataset size is {len(samples)}')
            print(f'num samples with more than one nonzero elem is {np.sum(np.count_nonzero(samples != 0, axis=1) > 1)}')
    
            # Print feature activation counts for comparison with limit case
            for feature in range(n_features):
                print(f'feature {feature}: {len(samples[samples[:, feature] != 0])}')
        return x
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

def generate_2d_ngon_vertices(n, rot=0., pad_to=None, force_length=0.9):
    # Angles for the vertices
    theta = torch.linspace(0, 2*np.pi, n+1)[:-1] + rot
    # Generate the vertices
    x = torch.cos(theta)
    y = torch.sin(theta)
    result = torch.stack((x, y))
    if pad_to is not None and n < pad_to:
        num_pad = pad_to - n
        result = torch.hstack([result, torch.zeros((2, num_pad))])
    return result * force_length

def generate_init_param(m, n):
    
    # Generate random angle and noise
    rand_angle = torch.rand(1).item() * 2*np.pi  # Uniform random angle
    noise = torch.randn(m, n) * 0.01  # Gaussian noise
    
    # Generate initialized parameters
    init_W = generate_2d_ngon_vertices(4, rot=rand_angle, pad_to=n) + noise
    init_b = torch.zeros(n)
    
    return init_W, init_b

class ToyModel(torch.nn.Module):
    def __init__(self, n_features, n_hidden, importance_decay=0.999):
        super().__init__()
        self.n_features = n_features
        W, b = self.initialize_params(n_hidden, n_features)
        #W, b = generate_init_param(n_hidden, n_features)
        self.W = torch.nn.Parameter(W)
        self.b = torch.nn.Parameter(b)
        # Generate importance values (power law)
        importances = (0.5 ** torch.arange(n_features))[None, :]
        #importances = torch.ones(n_features)
        self.register_buffer('importances', importances)
    
    def initialize_params(self, r=2, c=6, polygon_init=False):
        
        if polygon_init:
            # Initialize W matrix 
            W = torch.zeros(r, c)
        
            # Set first 4 columns to form a 4-gon
            W[:, 0] = torch.tensor([0.5, 0.5])
            W[:, 1] = torch.tensor([-0.5, 0.5])
            W[:, 2] = torch.tensor([-0.5, -0.5])
            W[:, 3] = torch.tensor([0.5, -0.5])
   
            # Initialize biases
            b = torch.zeros(c)
            b[4:] = -0.1  # negative values for remaining biases
   
            # Add Gaussian noise
            W += torch.randn_like(W) * 0.01
            b += torch.randn_like(b) * 0.01
        else:
            # Use Kaiming/He initialization
            #W = torch.nn.init.kaiming_uniform_(torch.empty(r, c))
            #b = torch.zeros(c)
            W = torch.randn(r, c) / np.sqrt(r)
            b = torch.zeros(c)

        return W, b
    
    def get_hidden_activations(self, x):
        return torch.matmul(x, self.W.T)
    
    def forward(self, x):
        h = self.get_hidden_activations(x)
        return torch.relu(torch.matmul(h, self.W) + self.b)

    def generate_batch(self, n_samples):
        # Initialize zero tensor to hold samples
        n_features = self.n_features
        x = np.zeros((n_samples, n_features))
        base = 2
    
        # Calculate the probabilities for each feature according to power law
        # The formula below ensures decreasing probabilities by feature index
        unnormalized_probs = np.array([1 / (base ** i) for i in range(n_features)])
        probs = unnormalized_probs / unnormalized_probs.sum()

        #probs = np.array([1 / n_features] * n_features)
    
        # Sample coordinates according to the power law distribution
        coords = np.random.choice(n_features, size=n_samples, p=probs)
    
        # Sample lengths uniformly
        lengths = np.random.uniform(0, 1, size=n_samples)
    
        # Put lengths in the right coordinates
        x[np.arange(n_samples), coords] = lengths
        
        return torch.tensor(x, dtype=torch.float)
    
def get_dataloader(sparsity, n_features=20, n_samples=50000, batch_size=512, limit=False, temperature=1.0):
    dataset = SparseRandomVectorDataset(n_samples, n_features, sparsity, limit, temperature)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def toy_model_loss(model, data):
    output = model(data)
    #return torch.mean((output - data) ** 2 * model.importances)
    return torch.mean((output - data) ** 2)

def train_toy_model(sparsity, n_features=20, n_hidden=5, n_dictionary=40, l1_lambda=0.01, epochs=100, log_freq=1, n_samples=50000, batch_size=512, limit=False, temperature=1.0, criterion=None):
    from sae import train_sparse_autoencoder, SparseAutoencoder, SAEEvaluator, visualize_sae_and_toy_model, select_top_k
    learning_rate = 0.001
    
    dataloader = get_dataloader(sparsity, n_features, n_samples, batch_size, limit, temperature)
    test_dataloader = get_dataloader(sparsity, n_features, 5000, 5000, limit, temperature)
    
    model = ToyModel(n_features, n_hidden)
    sae = SparseAutoencoder(n_hidden, n_dictionary, l1_lambda=l1_lambda)
    evaluator = SAEEvaluator(sae, model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sae_optimizer = torch.optim.AdamW(sae.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    models_saved = []
    saes_saved = []
    disentangled_models_saved = []
    mmcs_values = []
    wmmcs_values = []
    similarity_scores = []
    frob_norms = []
    permutation_distances = []

    save_path = f"results/training_run"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Move model to device
    model = model.to(device)
    sae = sae.to(device)

    epochs_sae_training = 100
    disable_tqdm = False
    start_again_each_epoch = False

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for batch in dataloader:
            #batch = model.generate_batch(batch_size).to(device)
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = toy_model_loss(model, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            '''
            #if epoch == 0:
            #   continue
            
            with torch.no_grad():
                hidden = model.get_hidden_activations(batch)
            sae_optimizer.zero_grad()
            sae_loss = sae.loss(hidden)
            sae_loss.backward()
            sae_optimizer.step()
            '''
            
            
            
            
            

        '''
        sae, sae_optimizer = train_sparse_autoencoder(model, sparsity, n_dictionary, l1_lambda, sae, sae_optimizer, epochs_sae_training, dataloader, disable_tqdm)

        results = evaluator.evaluate(test_dataloader)

        if results['mean_max_cs'] >= .999:
            epochs_sae_training = 10
            disable_tqdm = True
            print(f"mmcs is {results['mean_max_cs']} and sae training steps are {epochs_sae_training}")
        elif results['mean_max_cs'] >= .9997:
            epochs_sae_training = 1
            print(f"mmcs is {results['mean_max_cs']} and sae training steps are {epochs_sae_training}")
            disable_tqdm = True
        else:
            sae = SparseAutoencoder(n_hidden, n_dictionary, l1_lambda=l1_lambda)
            sae = sae.to(device)
            evaluator = SAEEvaluator(sae, model)
            sae_optimizer = torch.optim.AdamW(sae.parameters(), lr=0.001)
                
         '''
            

        if epoch % log_freq == 0:
            '''
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    batch = batch.to(device)
                    loss = toy_model_loss(model, batch)
                    test_loss += loss.item()
            '''
            
            train_losses.append(torch.tensor(train_loss / len(dataloader)))
            #test_losses.append(torch.tensor(test_loss / len(test_dataloader)))

            
            
            step = 0

            while True:
                # skip this loop while working on toy model
                #break

                if epoch > 4 and epoch < 10:
                    epochs_sae_training = 10
                elif epoch >= 10:
                    epochs_sae_training = 2
                    
                    
                
                if start_again_each_epoch:
                    sae = SparseAutoencoder(n_hidden, n_dictionary, l1_lambda=l1_lambda)
                    sae = sae.to(device)
                    sae_optimizer = torch.optim.AdamW(sae.parameters(), lr=0.001)
                    evaluator = SAEEvaluator(sae, model)
                
                sae, sae_optimizer = train_sparse_autoencoder(model, evaluator, sparsity, n_dictionary, l1_lambda, sae, sae_optimizer, epochs_sae_training, dataloader, disable_tqdm)
                
                toy_features = model.W.data.T
                sae_features = sae.encoder.weight.data
                sae_features = select_top_k(sae_features, n_features)
                sae_idx, similarity_score = evaluator.align_features(toy_features, sae_features)
                if similarity_score >= 0.9:
                    print(f'achieved good fit for SAE with {similarity_score}')
                    break
                if step == 0:
                    #print(f'tried too many times to find good fit for SAE, score is now {similarity_score}')
                    break
                step += 1
            
            
            
            toy_features = model.W.data.T
            sae_features = sae.encoder.weight.data
            sae_features = select_top_k(sae_features, n_features)
            sae_idx, similarity_score = evaluator.align_features(toy_features, sae_features)
            results = evaluator.evaluate(test_dataloader)
            WTW_sae = torch.matmul(sae_features, sae_features.T)
            WTW_toy = torch.matmul(toy_features, toy_features.T)
            frob_norms.append(torch.norm(WTW_toy - WTW_sae, p='fro'))
            
            models_saved += [copy.deepcopy(model)]
            saes_saved += [copy.deepcopy(sae)]
            mmcs_values.append(results['mean_max_cs'])
            wmmcs_values.append(results['weighted_mean_max_cs'])
            similarity_scores.append(similarity_score)
            permutation_distances.append(results['permutation_dist'])
            disentangled_models_saved.append(results['disentangled_model'])
            W_sae = sae.encoder.weight.data
            W_sae = select_top_k(W_sae, n_features)
            W_toy = model.W.data.T
            visualize_sae_and_toy_model(W_sae, W_toy, sparsity, epoch, save_path)
    # visualize_model(trained_model)
    return train_losses, test_losses, model, sae, models_saved, saes_saved, disentangled_models_saved, dataloader, mmcs_values, wmmcs_values, similarity_scores, frob_norms, permutation_distances

def visualize_model(model):
    W = model.W.detach().numpy()
    WTW = np.matmul(W.T, W)
    b = model.b.detach().numpy()
    width_ratios = [W.shape[1]//4, 1]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': width_ratios})
 
    sns.heatmap(WTW, ax=ax1, cmap='coolwarm', center=0, square=True, cbar=True, xticklabels=False, yticklabels=False)
    ax1.set_title("$W^T W$", fontsize=32)
    
    sns.heatmap(b.reshape(-1, 1), ax=ax2, cmap='coolwarm', center=0, cbar=True, xticklabels=False, yticklabels=False)
    ax2.set_title("$b$", fontsize=32)
    
    plt.tight_layout()
    plt.show()

def load_model(sparsity, n_features=20, n_hidden=5):
    try:
        model_state_dict = torch.load(f"results/toy_model_s{sparsity}_f{n_features}_h{n_hidden}.pth")
    except FileNotFoundError:
        print(f"Model for sparsity {sparsity} with {n_features} features and {n_hidden} hidden dimensions not found. Please run the experiment first.")
        return None
    model = ToyModel(n_features, n_hidden)
    model.load_state_dict(model_state_dict)
    return model

# %%
if __name__ == '__main__':
    n_features = 20
    n_hidden = 5
    for sparsity in [0.0, 0.7, 0.9, 0.97, 0.99, 0.997, 0.999]:
        print(f"Sparsity: {sparsity}")
        model = train_toy_model(sparsity, n_features, n_hidden)
        print("\n")

        # save model
        # torch.save(model.state_dict(), f"../../results/2024-11-14 measure superposition/toy_model_s{sparsity}_f{n_features}_h{n_hidden}.pth")

# %%
    # model = load_model(0.7, n_features, n_hidden)
    visualize_model(model)



