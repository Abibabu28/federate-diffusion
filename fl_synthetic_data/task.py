import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from datasets import load_dataset

from collections import OrderedDict

df=None
partitioner=None
class EmbedBlock(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedBlock, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Unflatten(1, (emb_dim,))
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class SinusoidalPositionEmbedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class epsilon_diffuse(nn.Module):
    def __init__(self, input_dim, output_dim, T, ns_layer=[10,5,3], t_embed_dim=8, c_embed_dim=2):
        super(epsilon_diffuse,self).__init__()
        self.T = T
        self.fc1=nn.Linear(input_dim, ns_layer[0], bias=True)
        self.fc2=nn.Linear(ns_layer[0], ns_layer[1], bias=True)
        self.fc3=nn.Linear(ns_layer[1], ns_layer[2], bias=True)
        self.fc4=nn.Linear(ns_layer[2], output_dim)

        self.sinusoidaltime = SinusoidalPositionEmbedBlock(t_embed_dim)
        self.t_emb1 = EmbedBlock(t_embed_dim, ns_layer[0])
        self.t_emb2 = EmbedBlock(t_embed_dim, ns_layer[1])
        self.c_embed1 = EmbedBlock(c_embed_dim, ns_layer[0])
        self.c_embed2 = EmbedBlock(c_embed_dim, ns_layer[1])

    def forward(self, x, t, c, c_mask):
        t = t.float() / self.T
        t = self.sinusoidaltime(t)
        t_emb1 = self.t_emb1(t)
        t_emb2 = self.t_emb2(t)

        c = c*c_mask
        c_emb1 = self.c_embed1(c)
        c_emb2 = self.c_embed2(c)

        out=torch.relu(self.fc1(x))
        out=torch.relu(self.fc2(c_emb1*out + t_emb1))
        out=torch.relu(self.fc3(c_emb2*out + t_emb2))
        out=self.fc4(out)
        return out


class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        n, d = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, d).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1) * eta
        return noisy

    def backward(self, x, t):
        return self.network(x, t)

    def backward_cfg(self, x, t, c, c_mask):
        return self.network(x, t, c, c_mask)


def get_context_mask(c, drop_prob, n_classes=2, device='cpu'):
    c_hot = F.one_hot(c.to(torch.int64), num_classes=n_classes).to(device)
    c_mask = torch.bernoulli(torch.ones_like(c_hot).float() - drop_prob).to(device)
    return c_hot, c_mask


def create_diffusion_model(input_dim, output_dim,device):
    """Create a diffusion model with specified input dimension."""
    # Model parameters
    # Set constants
    T_col = 8
    T_row = 100
    T = T_col*T_row
   
    n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02
    # Initialize model
    model = MyDDPM(
        epsilon_diffuse(
            input_dim, 
            output_dim, 
            T,
            ns_layer=[10,5,3],
            c_embed_dim=2
        ),
        n_steps=n_steps, 
        min_beta=min_beta, 
        max_beta=max_beta, 
        device=device
    )
    
    return model


def get_weights(model):
    """Extract model weights as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

"""def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)"""


def set_weights(model, parameters):
    """
    Load model weights from a saved state dictionary for federated learning.
    """
    print("inside set weights")
    store_path = "ddpm_model.pt"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    # Load the state dictionary from the specified path
    model.load_state_dict(torch.load(store_path, map_location=device))
    


def clean_and_prepare_data(df, name):
    print(f"\nCleaning and preparing {name} dataset...")

    # Create array X
    features_to_check = [col for col in df.columns if col != 'event_status']
    X = df[features_to_check].values
    y = df['event_status'].values

    # Check for issues
    n_missing = np.isnan(X).sum()
    n_zeros = (X == 0).sum()
    n_negative = (X < 0).sum()

    if n_missing > 0 or n_zeros > 0 or n_negative > 0:
        print(f"Found issues in {name} dataset:")
        print(f"Missing values: {n_missing}")
        print(f"Zero values: {n_zeros}")
        print(f"Negative values: {n_negative}")

    # Create a mask to find rows where time_to_event is not 0
    # Assuming time_to_event is the last column, adjust if needed
    if 'time_to_event' in features_to_check:
        time_col_idx = features_to_check.index('time_to_event')
    else:
        # If column not found, use the last column as fallback
        time_col_idx = -1
        
    mask = X[:, time_col_idx] != 0

    # Apply the mask to remove rows with zero time_to_event
    X_clean = X[mask]
    y_clean = y[mask]

    print(f"Original shape: {X.shape}")
    print(f"Cleaned shape: {X_clean.shape}")
    print(f"Event rate: {y_clean.mean():.3f}")

    return X_clean, y_clean, features_to_check


def training_loop_cfg(ddpm, data, labels, n_epochs, optim, device, n_classes=2, c_drop_prob=0.1, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        x0 = data
        n = len(x0)
        c_hot, c_mask = get_context_mask(labels, c_drop_prob, n_classes=n_classes, device=device)

        eta = torch.randn_like(x0).to(device)
        t = torch.randint(0, n_steps, (n,)).to(device)

        noisy_imgs = ddpm(x0, t, eta)
        eta_theta = ddpm.backward_cfg(noisy_imgs, t.reshape(n, -1), c_hot, c_mask)

        loss = mse(eta_theta, eta)
        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_loss += loss.item() * len(x0) / len(data)

        if epoch % 10 == 0:
            print(f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}")

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            if epoch % 10 == 0:
                print(f"Best model saved (loss: {epoch_loss:.3f})")
    
    return best_loss

def load_data(partition_id: int, num_partitions: int):
    global df
    global partitioner
    
    if df is None or partitioner is None:
        print(f"Loading dataset and creating {num_partitions} partitions...")
        # Load the dataset
        dataset = load_dataset("csv", data_files="clean_alcohol_df.csv")
        
        # Create partitioner (you'll need to implement or import IidPartitioner)
        from flwr_datasets.partitioner import IidPartitioner
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset["train"]
        print(f"Dataset loaded with {len(dataset['train'])} rows")
    
    # Get the partition for this client
    partition = partitioner.load_partition(partition_id=partition_id)
    
    # Convert to pandas DataFrame
    df_partition = partition.to_pandas()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    etiology="alcohol"
    
   
    X_clean, y_clean, features = clean_and_prepare_data(df_partition, etiology)
    
    

    # Convert to torch tensors and move to device
    labels = torch.tensor(y_clean).float().to(device)
    data = torch.log(torch.tensor(X_clean).float().to(device))

    print(f"Prepared data shape: {data.shape}")
    print(f"Prepared labels shape: {labels.shape}")

    # Model parameters
    
    input_dim = data.shape[1]
   
    output_dim = data.shape[1]
    

    # Initialize model
    print(f"\nInitializing model for {etiology}...")
    
    
    return input_dim, output_dim, data,labels,features,X_clean, y_clean

def generate_new_data_cfg(ddpm, n_samples, labels, n_classes=2, device=None, frames_per_gif=100, d=2, w_val=0.):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, d).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            c_drop_prob = 0
            c_hot, c_mask = get_context_mask(labels, c_drop_prob, n_classes=n_classes, device=device)
            eta_theta_keep_class = ddpm.backward_cfg(x, time_tensor, c_hot, c_mask)
            c_mask = torch.zeros_like(c_mask)
            eta_theta_drop_class = ddpm.backward_cfg(x, time_tensor, c_hot, c_mask)
            eta_theta = (1 + w_val) * eta_theta_keep_class - w_val * eta_theta_drop_class

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, d).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

    return x
