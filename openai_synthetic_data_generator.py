import os
from openai import OpenAI
import pandas as pd
import numpy as np
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from io import StringIO
from key import OPENAI_API_KEY

class SyntheticDataGenerator:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        api_key = OPENAI_API_KEY
        
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.output_dir = Path("generated_data")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_data(self, total_samples=30000, batch_size=300):
        """
        Generate synthetic data purely from prompt.
        """
        output_file = self.output_dir / f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print(f"Generating {total_samples} synthetic samples...")
        
        # Generate all synthetic samples from scratch
        synthetic_data = self._generate_samples(total_samples, batch_size)
        
        # Save the final dataset
        synthetic_data.to_csv(output_file, index=False)
        print(f"Final dataset with {len(synthetic_data)} samples saved to {output_file}")
        
        # Print statistics
        self._print_dataset_statistics(synthetic_data)
        
        return synthetic_data
    
    def _generate_samples(self, num_samples, batch_size):
        """Generate synthetic samples using OpenAI."""
        print(f"Generating {num_samples} synthetic samples...")
        
        # Process in batches
        data_list = []
        remaining_samples = num_samples
        
        while remaining_samples > 0:
            batch_size_actual = min(batch_size, remaining_samples)
            print(f"Generating batch of {batch_size_actual} samples ({remaining_samples} remaining)...")
            
            prompt = """
            Generate {num_samples} synthetic observational data points using the following data-generating process, which includes unmeasured confounding.

            Data Generating Process:
            1. Generate covariate X from a uniform distribution between -1.5 and 1.5.
            2. Introduce an unobserved confounder U from a standard normal distribution.
            3. Treatment assignment (A) depends on both X and U:
               - Use a logistic function: P(A=1 | X, U) = sigmoid(α * X + β * U + c)
               - Choose α, β such that X has a moderate effect and U has a strong confounding effect.
            4. The outcome (y) depends nonlinearly on X, U, and A:
               - Compute a set of nonlinear features of X (e.g., sinusoids, exponentials, interactions)
               - y0 = f(X, U): baseline outcome under control
               - Treatment effect varies nonlinearly with X
               - y1 = y0 + Δ(X): treated outcome
               - Observed outcome y = A * y1 + (1 - A) * y0
            5. Add heteroskedastic noise to y:
               - Noise variance increases with a nonlinear function of X or U

            Key Properties:
            - Unmeasured confounding is present through U affecting both A and y
            - Treatment effect is heterogeneous and nonlinearly related to X
            - Average treatment effect is positive (~15 units), i.e., treatment increases y
            - The outcome y is continuous
            - Maintain complex nonlinear relationships between variables

            Return ONLY a CSV with columns: X, A, y
            """.format(num_samples=batch_size_actual)
            
            # Generate batch of synthetic data
            response = self._call_openai_for_synthetic_data(prompt, batch_size_actual)
            data_list.append(response)
            remaining_samples -= len(response)
        
        # Combine all batches
        synthetic_data = pd.concat(data_list, ignore_index=True)
        
        print(f"Successfully generated {len(synthetic_data)} synthetic samples")
        print(f"Data average outcome: {synthetic_data['y'].mean():.2f}")
        
        return synthetic_data
    
    def _call_openai_for_synthetic_data(self, prompt, batch_size, max_retries=3):
        """Call OpenAI API to generate complete synthetic data with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates synthetic data according to complex data generating processes."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                
                # Extract content from response
                content = response.choices[0].message.content
                
                # Save raw response for debugging
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_file = self.output_dir / f"raw_synth_response_{timestamp}.txt"
                with open(raw_file, "w") as f:
                    f.write(content)
                print(f"Raw synthetic response saved to {raw_file}")
                
                # Clean up response if needed
                if "```" in content:
                    parts = content.split("```")
                    for part in parts:
                        if "," in part and "X" in part and "A" in part:
                            content = part.strip()
                            if content.startswith("csv") or content.startswith("CSV"):
                                content = content[3:].strip()
                            break
                
                # Parse CSV using StringIO
                df = pd.read_csv(StringIO(content))
                
                # Standardize column names
                column_mapping = {}
                for col in df.columns:
                    if col.lower() == 'x': column_mapping[col] = 'X'
                    elif col.lower() == 'a': column_mapping[col] = 'A'
                    elif col.lower() == 'y': column_mapping[col] = 'y'
                
                if column_mapping:
                    df = df.rename(columns=column_mapping)
                
                # Ensure we have all required columns
                required_cols = ['X', 'A', 'y']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns. Got: {df.columns.tolist()}")
                
                # Keep only the required columns
                df = df[required_cols]
                
                # Make sure X is in the right range
                X_min, X_max = -1.5, 1.5
                if df['X'].min() < X_min or df['X'].max() > X_max:
                    print(f"Warning: X values out of expected range [{X_min}, {X_max}]. Clipping values.")
                    df['X'] = df['X'].clip(X_min, X_max)
                
                # Ensure A is binary
                df['A'] = df['A'].astype(int)
                
                # Ensure we have the right number of samples (or fewer)
                if len(df) > batch_size:
                    df = df.iloc[:batch_size]
                
                # If we get fewer samples than requested, try to generate the remaining
                if len(df) < batch_size:
                    remaining = batch_size - len(df)
                    print(f"Only received {len(df)} samples, generating {remaining} more...")
                    
                    remaining_prompt = """
                    Generate {num_samples} synthetic observational data points using the following data-generating process, which includes unmeasured confounding.

                    Data Generating Process:
                    1. Generate covariate X from a uniform distribution between -1.5 and 1.5.
                    2. Introduce an unobserved confounder U from a standard normal distribution.
                    3. Treatment assignment (A) depends on both X and U:
                       - Use a logistic function: P(A=1 | X, U) = sigmoid(α * X + β * U + c)
                       - Choose α, β such that X has a moderate effect and U has a strong confounding effect.
                    4. The outcome (y) depends nonlinearly on X, U, and A:
                       - Compute a set of nonlinear features of X (e.g., sinusoids, exponentials, interactions)
                       - y0 = f(X, U): baseline outcome under control
                       - Treatment effect varies nonlinearly with X
                       - y1 = y0 + Δ(X): treated outcome
                       - Observed outcome y = A * y1 + (1 - A) * y0
                    5. Add heteroskedastic noise to y:
                       - Noise variance increases with a nonlinear function of X or U

                    Key Properties:
                    - Unmeasured confounding is present through U affecting both A and y
                    - Treatment effect is heterogeneous and nonlinearly related to X
                    - Average treatment effect is positive (~15 units), i.e., treatment increases y
                    - The outcome y is continuous
                    - Maintain complex nonlinear relationships between variables

                    Return ONLY a CSV with columns: X, A, y
                    """.format(num_samples=remaining)
                    
                    try:
                        additional_response = self._call_openai_for_synthetic_data(remaining_prompt, remaining)
                        df = pd.concat([df, additional_response], ignore_index=True)
                    except Exception as e:
                        print(f"Failed to generate additional samples: {e}")
                
                return df
            
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("All retry attempts failed.")
                    raise RuntimeError(f"Failed to generate synthetic data after {max_retries} attempts: {e}")
    
    def _print_dataset_statistics(self, data):
        """Print key statistics about the generated dataset."""
        print("\n===== Dataset Statistics =====")
        print(f"Total samples: {len(data)}")
        
        # Overall statistics
        print("\nOverall statistics:")
        print(f"Mean X: {data['X'].mean():.4f}")
        print(f"X range: [{data['X'].min():.4f}, {data['X'].max():.4f}]")
        print(f"Treatment rate: {data['A'].mean():.2%}")
        print(f"Mean outcome: {data['y'].mean():.4f}")
        
        # Treatment group statistics
        treated = data[data['A'] == 1]
        control = data[data['A'] == 0]
        print("\nTreatment group statistics:")
        print(f"Number of treated units: {len(treated)} ({len(treated)/len(data):.2%})")
        print(f"Treated mean outcome: {treated['y'].mean():.4f}")
        print(f"Control mean outcome: {control['y'].mean():.4f}")
        print(f"Naive treatment effect: {treated['y'].mean() - control['y'].mean():.4f}")
        
        # Covariate distribution by group
        print("\nCovariate distribution by treatment group:")
        print(f"Mean X in treated group: {treated['X'].mean():.4f}")
        print(f"Mean X in control group: {control['X'].mean():.4f}")
        
        # Correlation between covariates and y
        print("\nRelationships:")
        print(f"Correlation between X and y: {data['X'].corr(data['y']):.4f}")
        print(f"Correlation between X and A: {data['X'].corr(data['A']):.4f}")
        print(f"Correlation between A and y: {data['A'].corr(data['y']):.4f}")
        
        # Check for nonlinearity using binned statistics
        print("\nNonlinear relationship check (binned statistics):")
        bins = pd.cut(data['X'], 10)
        bin_means = data.groupby(bins)['y'].mean()
        bin_treat_rate = data.groupby(bins)['A'].mean()
        
        print("X bin\t\tMean y\t\tTreatment rate")
        for bin_name, mean_y, treat_rate in zip(bin_means.index, bin_means, bin_treat_rate):
            print(f"{bin_name}\t{mean_y:.4f}\t{treat_rate:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data purely from prompts")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--num_samples", type=int, default=30000, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size for API calls")
    parser.add_argument("--output", type=str, default="optimized_generated_data.csv", help="Output file name")
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(api_key=args.api_key, model=args.model)
    
    # Generate data
    synthetic_data = generator.generate_data(total_samples=args.num_samples, batch_size=args.batch_size)
    
    # Also save to the specific output file if provided
    if args.output:
        output_path = Path(args.output)
        synthetic_data.to_csv(output_path, index=False)
        print(f"Dataset also saved to {output_path}")