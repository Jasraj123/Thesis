import numpy as np
import pandas as pd
from scipy.special import expit

def generate_observational_data(n_samples=100000, seed=42):
    """
    Generate observational data with a single covariate:
    
    Covariates:
    - X ~ Uniform(-1, 1)
    
    Treatment assignment (linear confounding):
    P(A=1|X) = expit(2*X)
    
    Outcome model (linear):
    Y = 2*X + 1.5*A + ε
    """
    np.random.seed(seed)
    
    # Generate single covariate
    X = np.random.uniform(-1, 1, size=n_samples)
    
    # Linear confounding based on X
    propensity = expit(2*X)
    
    # Set seed again before generating treatment to match pattern
    np.random.seed(seed + 100)  # Use a different offset to match the pattern
    A = np.random.binomial(n=1, p=propensity)
    
    # Set seed again before generating noise to match pattern
    np.random.seed(seed + 200)  # Use another offset
    base_noise = np.random.normal(0, 0.8, size=n_samples)
    
    # Linear outcome model
    Y = 2*X + 1.5*A + base_noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'X': X,
        'A': A,
        'y': Y
    })
    
    return df

if __name__ == "__main__":
    # Generate a large observational dataset
    obs_data = generate_observational_data(n_samples=250000, seed=42)
    
    # Save to CSV
    obs_data.to_csv("optimized_generated_data.csv", index=False)
    print("Generated observational data saved to optimized_generated_data.csv")
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Number of samples: {len(obs_data)}")
    print(f"Treatment proportion: {obs_data['A'].mean():.3f}")
    print("\nSummary statistics:")
    print(obs_data.describe())
