"""fl-synthetic-data: A Flower / PyTorch app."""

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_synthetic_data.task import (
    get_weights,set_weights, load_data, set_weights,training_loop_cfg, create_diffusion_model,get_context_mask,
    generate_new_data_cfg
)
from typing import Dict, List, Tuple, Union, Optional



# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, input_dim, output_dim, data,labels, features,x_clean, y_clean,local_epochs):
       
        self.data = data
        self.labels = labels
        self.features=features
        self.local_epochs = local_epochs
        self.x_clean= x_clean
        self.y_clean= y_clean
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        
        # Initialize diffusion model for synthetic data generation
        self.input_dim=input_dim
        self.ddpm = create_diffusion_model(input_dim, output_dim,self.device)

    def generate_synthetic_data(self):
        """Generate synthetic data using the trained model."""
        self.ddpm.eval()

        # Calculate samples needed
        original_size = len(self.x_clean)
        original_event_rate = self.y_clean.mean()
        n_events_needed = int(original_size * original_event_rate)
        n_nonevents_needed = original_size - n_events_needed
        
        print(f"Events needed: {n_events_needed}")
        print(f"Non-events needed: {n_nonevents_needed}")

        # Generate synthetic events
        gen_labels_events = torch.ones(n_events_needed).to(self.device)
        generated_events = generate_new_data_cfg(
            self.ddpm, n_events_needed, gen_labels_events,
            n_classes=2, device=self.device, d=self.input_dim
        )

        # Generate synthetic non-events
        gen_labels_nonevents = torch.zeros(n_nonevents_needed).to(self.device)
        generated_nonevents = generate_new_data_cfg(
            self.ddpm, n_nonevents_needed, gen_labels_nonevents,
            n_classes=2, device=self.device, d=self.input_dim
        )

        df_gen_events = pd.DataFrame(np.exp(generated_events.cpu()), columns=self.features)
        df_gen_nonevents = pd.DataFrame(np.exp(generated_nonevents.cpu()), columns=self.features)

         # Add event_status
        df_gen_events['event_status'] = 1
        df_gen_nonevents['event_status'] = 0

        # Combine and shuffle
        df_gen = pd.concat([df_gen_events, df_gen_nonevents], ignore_index=True)
        df_gen = df_gen.sample(frac=1, random_state=42).reset_index(drop=True)

        df_gen.to_csv("synthetic.csv", index=False)



     
    def fit(self, parameters, config):
        set_weights (self.ddpm ,parameters)
        
        
        # Train the diffusion model
        optimizer = optim.Adam(self.ddpm.parameters(), lr=0.001)
        train_loss = training_loop_cfg(
            self.ddpm, 
            self.data, 
            self.labels, 
            n_epochs=self.local_epochs,
            optim=optimizer, 
            device=self.device, 
            n_classes=2, 
            c_drop_prob=0.1, 
            store_path="ddpm_model.pt"
        )
        
        self.generate_synthetic_data()
        
        

        return (
            get_weights(self.ddpm),
            len(self.data),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters: List[np.ndarray],config):
        """Evaluate the model on local data."""
        set_weights(self.ddpm,parameters)
        
        # Evaluation
        self.ddpm.eval()
        mse = nn.MSELoss()
        
        with torch.no_grad():
            n = len(self.data)
            c_hot, c_mask = get_context_mask(self.labels, 0.0, n_classes=2, device=self.device)
            
            t = torch.randint(0, self.ddpm.n_steps, (n,)).to(self.device)
            eta = torch.randn_like(self.data).to(self.device)
            
            noisy_imgs = self.ddpm(self.data, t, eta)
            eta_theta = self.ddpm.backward_cfg(noisy_imgs, t.reshape(n, -1), c_hot, c_mask)
            
            loss = mse(eta_theta, eta)
        
        
        # Return metrics
        return loss.item(), n, {"loss": loss.item()}


def client_fn(context: Context):
    # Load model and data
   
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    input_dim, output_dim, data,labels,features,x_clean, y_clean = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    

    # Return Client instance
    return FlowerClient(input_dim, output_dim, data,labels,features,x_clean, y_clean,local_epochs).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)
   
