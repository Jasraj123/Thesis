import random
import sys, os, yaml
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy import interpolate
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class PropensityNN(nn.Module):
    """Neural network for propensity score estimation"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super(PropensityNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class OutcomeNN(nn.Module):
    """Neural network for outcome estimation"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super(OutcomeNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class DRNeuralNetEstimator:
    """
    Doubly Robust Neural Network Estimator 
    (replacement for ForestDRLearner)
    """
    def __init__(self, hidden_dims=[64, 32], lr=0.001, epochs=100, batch_size=32):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.propensity_model = None
        self.outcome_model = None
    
    def fit(self, y, t, X):
        # Convert to PyTorch tensors and scale features
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        t_tensor = torch.FloatTensor(t).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loaders
        dataset = TensorDataset(X_tensor, t_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize models
        input_dim = X.shape[1]
        self.propensity_model = PropensityNN(input_dim, self.hidden_dims).to(self.device)
        self.outcome_model = OutcomeNN(input_dim + 1, self.hidden_dims).to(self.device)
        
        # Initialize optimizers
        prop_optimizer = optim.Adam(self.propensity_model.parameters(), lr=self.lr)
        outcome_optimizer = optim.Adam(self.outcome_model.parameters(), lr=self.lr)
        
        # Loss functions
        prop_criterion = nn.BCELoss()
        outcome_criterion = nn.MSELoss()
        
        # Train propensity model
        self.propensity_model.train()
        for epoch in range(self.epochs):
            prop_epoch_loss = 0
            for X_batch, t_batch, _ in dataloader:
                # Forward pass
                prop_pred = self.propensity_model(X_batch).squeeze()
                
                # Calculate loss
                prop_loss = prop_criterion(prop_pred, t_batch.squeeze())
                
                # Backward pass
                prop_optimizer.zero_grad()
                prop_loss.backward()
                prop_optimizer.step()
                
                prop_epoch_loss += prop_loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Propensity Epoch [{epoch+1}/{self.epochs}], Loss: {prop_epoch_loss/len(dataloader):.4f}')
        
        # Train outcome model
        self.outcome_model.train()
        for epoch in range(self.epochs):
            outcome_epoch_loss = 0
            for X_batch, t_batch, y_batch in dataloader:
                # Concatenate treatment with features
                outcome_input = torch.cat([X_batch, t_batch.unsqueeze(1)], dim=1)
                
                # Forward pass
                outcome_pred = self.outcome_model(outcome_input).squeeze()
                
                # Calculate loss
                outcome_loss = outcome_criterion(outcome_pred, y_batch.squeeze())
                
                # Backward pass
                outcome_optimizer.zero_grad()
                outcome_loss.backward()
                outcome_optimizer.step()
                
                outcome_epoch_loss += outcome_loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Outcome Epoch [{epoch+1}/{self.epochs}], Loss: {outcome_epoch_loss/len(dataloader):.4f}')
        
        return self
    
    def effect(self, X):
        """Estimate CATE for each sample"""
        if self.propensity_model is None or self.outcome_model is None:
            raise ValueError("Models not fitted yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Create treatment=1 inputs
        X_t1 = torch.cat([X_tensor, torch.ones(X_tensor.shape[0], 1).to(self.device)], dim=1)
        
        # Create treatment=0 inputs
        X_t0 = torch.cat([X_tensor, torch.zeros(X_tensor.shape[0], 1).to(self.device)], dim=1)
        
        # Get predictions
        self.outcome_model.eval()
        with torch.no_grad():
            y1_pred = self.outcome_model(X_t1).cpu().numpy()
            y0_pred = self.outcome_model(X_t0).cpu().numpy()
        
        # Calculate CATE
        cate = y1_pred - y0_pred
        return cate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.parent.absolute())


def load_yaml(path_relative):
    return yaml.safe_load(open(get_project_path() + path_relative + ".yaml", 'r'))

def load_all_yaml_from_directory(path_relative):
    path = get_project_path() + path_relative
    files = os.listdir(path)
    configs = []
    for file in files:
        if file.endswith(".yaml"):
            configs.append(load_yaml(path_relative + file[:-5]))
    return configs

def save_yaml(path_relative, file):
    with open(get_project_path() + path_relative + ".yaml", 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)

def estimate_e(X, A, hidden_dims=[64, 32], epochs=100, batch_size=32):
    '''
    Estimate propensity score using a neural network model
    '''
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    A_tensor = torch.FloatTensor(A).to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, A_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_dim = X.shape[1]
    model = PropensityNN(input_dim, hidden_dims).to(device)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, A_batch in dataloader:
            # Forward pass
            pred = model(X_batch).squeeze()
            
            # Calculate loss
            loss = criterion(pred, A_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Propensity Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
    
    # Get propensity scores
    model.eval()
    with torch.no_grad():
        e = model(X_tensor).cpu().numpy()
    
    return e


def estimate_mu(X, A, y, hidden_dims=[64, 32], epochs=100, batch_size=32):
    '''
    Estimate response function using a neural network model
    '''
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    A_tensor = torch.FloatTensor(A).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    # Create outcome model input (X concatenated with A)
    train_data = torch.cat([X_tensor, A_tensor.unsqueeze(1)], dim=1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(train_data, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_dim = X.shape[1] + 1  # +1 for treatment indicator
    model = OutcomeNN(input_dim, hidden_dims).to(device)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            # Forward pass
            pred = model(X_batch).squeeze()
            
            # Calculate loss
            loss = criterion(pred, y_batch.squeeze())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Outcome Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
    
    # Prepare test data for A=0 and A=1
    test_0 = torch.cat([X_tensor, torch.zeros(X_tensor.shape[0], 1).to(device)], dim=1)
    test_1 = torch.cat([X_tensor, torch.ones(X_tensor.shape[0], 1).to(device)], dim=1)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        mu0 = model(test_0).cpu().numpy()
        mu1 = model(test_1).cpu().numpy()
    
    return mu0, mu1


def get_estimates(dataset_train, dataset_val, delta, significance_level=0.05):
    '''Param setting'''
    alpha = significance_level
    z_alpha = norm.ppf(1 - alpha/2)
    Y_train = np.array(dataset_train['y']).reshape(-1, 1)
    A_train = np.array(dataset_train['A']).reshape(-1, 1)
    X_train = np.array(dataset_train['X']).reshape(-1, 1)
    n = Y_train.shape[0]
    
    print("Training neural network models...")
    
    '''IPW Implementation test'''
    print("Estimating propensity scores...")
    e = estimate_e(X_train, A_train.flatten(), epochs=50)
    
    print("Estimating outcome models...")
    mu0, mu1 = estimate_mu(X_train, A_train.flatten(), Y_train.flatten(), epochs=50)
    
    print("Calculating AIPW estimator...")
    aipw = (A_train * Y_train / e - (1 - A_train) * Y_train / (1 - e)) - \
            ((A_train - e) / e * (1 - e)) * ((1-e) * mu1 + e * mu0)
    ate_est_aipw = np.mean(aipw)
    ate_ci_aipw = (ate_est_aipw - z_alpha * np.sqrt(np.var(aipw)/n), ate_est_aipw + z_alpha * np.sqrt(np.var(aipw)/n))

    print("var_aipw", np.var(aipw))
    print("ate_est_aipw", ate_est_aipw)
    print("ate_ci_aipw", ate_ci_aipw)

    '''PPI Implementation'''
    print("Implementing PPI procedure...")
    N = np.array(dataset_val['y']).shape[0]
    N_train = int(N/2)
    N_eval = N - N_train

    Y_N_train = np.array(dataset_val['y']).reshape(-1, 1)[:N_train]
    T_N_train = np.array(dataset_val['A']).reshape(-1, 1)[:N_train]
    X_N_train = np.array(dataset_val['X']).reshape(-1, 1)[:N_train]
    
    X_N_eval = np.array(dataset_val['X']).reshape(-1, 1)[N_train:]
    T_N_eval = np.array(dataset_val['A']).reshape(-1, 1)[N_train:]
    Y_N_eval = np.array(dataset_val['y']).reshape(-1, 1)[N_train:]

    print("Training Doubly Robust Neural Network Estimator...")
    est_nn = DRNeuralNetEstimator(
        hidden_dims=[64, 32],
        lr=0.001,
        epochs=50,
        batch_size=32
    )
    
    y_N_train = Y_N_train.reshape(N_train)
    est_nn.fit(y_N_train, T_N_train.flatten(), X_N_train)

    cate_N = est_nn.effect(X_N_eval)
    ate_N = np.mean(cate_N)
    var_N = np.var(cate_N)
    pred_n = est_nn.effect(X_train)

    mean_rectifier = np.mean(aipw - pred_n)
    var_rectifier = np.var(aipw - pred_n)
    
    ate_est_ppi = ate_N + mean_rectifier
    ate_ci_norm_ppi = (ate_est_ppi - z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval),
                       ate_est_ppi + z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval))
    print("var_ppi", var_rectifier)
    print("ate_est_ppi", ate_est_ppi)
    print("ate_ci_ppi", ate_ci_norm_ppi)

    '''Normal/Asymptotic setting: Observational data only'''
    ate_est_obs = ate_N
    ate_ci_obs = (ate_est_obs - z_alpha * np.sqrt(var_N/N_eval), \
                  ate_est_obs + z_alpha * np.sqrt(var_N/N_eval))
    print("ate_ci_obs", ate_ci_obs)

    return [ate_est_aipw, ate_est_ppi, ate_est_obs], \
           [ate_ci_aipw, ate_ci_norm_ppi, ate_ci_obs]

def sim_cases(seed, df_rct, df_obs, significance_level, delta):
    """
    Run simulation cases with provided RCT and observational datasets.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    df_rct : pandas.DataFrame
        RCT dataset
    df_obs : pandas.DataFrame
        Observational dataset
    significance_level : float
        Statistical significance level (alpha)
    delta : float
        Parameter for confidence interval calculations
    
    Returns:
    --------
    ate_estimates : list
        List of ATE estimates from different methods
    ate_ci : list
        List of confidence intervals for ATE estimates
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Get estimates
    ate_estimates, ate_ci = get_estimates(df_rct, df_obs, delta, significance_level)

    return ate_estimates, ate_ci 