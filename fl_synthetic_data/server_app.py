"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

import torch
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context, NDArrays, Parameters

from fl_synthetic_data.task import MyDDPM, epsilon_diffuse,get_weights,create_diffusion_model




"""def get_initial_parameters() -> Parameters:
    #Initialize model parameters for the global model.
    # Create a dummy model to get initial parameters
    input_dim = 9  # Number of features in your dataset (adjust if needed)
    T_col = 8
    T_row = 25
    T = T_col * T_row
    
    # Create network
    network = epsilon_diffuse(
        input_dim=input_dim, 
        output_dim=input_dim, 
        T=T, 
        ns_layer=[10, 5, 3],  # Simplified architecture
        t_embed_dim=8, 
        c_embed_dim=2
    )
    
    # Create DDPM
    ddpm = MyDDPM(
        network=network,
        n_steps=200,
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    
    # Get weights as numpy arrays
    weights = [val.cpu().numpy() for _, val in ddpm.state_dict().items()]
    
    return ndarrays_to_parameters(weights)"""

def weighted_average(metrics):
    """Aggregate metrics from clients."""
    # Extract loss values and weights
    if not metrics:
        return {}
    
    losses = [num_examples * m.get("loss", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Compute weighted average
    if sum(examples) == 0:
        return {"loss": 0.0}
    
    return {"loss": sum(losses) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Create and return server components for federated learning."""
    # Get initial parameters
    

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    
    # Get input dimensions from config or use a default
    input_dim = context.run_config.get("input_dim",9)  
    output_dim = context.run_config.get("output_dim",9) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize model parameters
    model = create_diffusion_model(input_dim=input_dim,output_dim=output_dim,device=device)
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Define strategy
    strategy = FedAvg(
        initial_parameters=parameters,
        fraction_evaluate=1.0,
        min_available_clients=2,
        fraction_fit=fraction_fit,
    )
    
    # Define server configuration
   
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)