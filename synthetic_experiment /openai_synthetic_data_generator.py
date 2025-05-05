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
    def __init__(self, api_key=None, model="gpt-4o"):
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
            
            prompt = f"""
                Generate {batch_size_actual} synthetic observational data points. Return a CSV with exactly three columns: X, A, y.

                Data generation steps:
                1. Sample covariate X ~ Uniform(-1.5, 1.5)
                2. Sample unobserved confounder U ~ Normal(0, 1.5)  # Do NOT include U in the output
                3. Compute additional covariates:
                - X1 = X + Normal(0, 0.1)
                - X2 = X^2 + Normal(0, 0.15)
                - X3 = sin(2 * X) + Normal(0, 0.2)
                4. Treatment assignment A ~ Bernoulli(p), where p = sigmoid(0.6 * X + 0.2 * U)
                5. Compute outcome components:
                - y0 = 3 + 2 * X + 1.5 * X1 - 2 * X2 + 1.8 * cos(X)
                        + 0.5 * X * X1 + 0.2 * X2 * X3
                        + 0.5 * (X > 0) * sin(5 * X)
                        + 1.1 * cos(X^2) + 0.6 * sin(exp(0.7 * X^2)) - 0.6 * U
                - treatment_effect = 3 + 1.5 * X + 2 * X2 + 0.7 * sin(1.5 * X)
                        - 1 * X3 + 0.6 * (X > 0.5) * tanh(3 * X)
                        + 0.8 * (X < -0.5) * tanh(4 * X)
                        + 0.4 * X * sin(2.5 * X) + Normal(0, 0.3) + 0.7 * U
                - y1 = y0 + treatment_effect
                6. Add heteroskedastic noise ε ~ Normal(0, noise_scale), where:
                noise_scale = 0.25 + 0.3 * abs(sin(2.5 * X)) + 0.4 * (X > 1.0) + 0.3 * abs(U)
                7. Final observed outcome: y = A * y1 + (1 - A) * y0 + ε

                Output ONLY the CSV with three columns in order: X,A,y

                IMPORTANT: Do not include any explanations or preamble. Begin your response with the CSV header: X,A,y
            """.format(num_samples=batch_size_actual)
            
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
                # Add explicit instructions to the prompt
                modified_prompt = prompt + "\n\nIMPORTANT: Start your response with ONLY the CSV header 'X,A,y' followed by data rows. Do not include any text descriptions or explanations."
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a data generator that outputs ONLY CSV data. Do not include any explanations or descriptions."},
                        {"role": "user", "content": modified_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                
                # Extract content and save for debugging
                content = response.choices[0].message.content
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_file = self.output_dir / f"raw_synth_response_{timestamp}.txt"
                with open(raw_file, "w") as f:
                    f.write(content)
                print(f"Raw synthetic response saved to {raw_file}")
                
                # Find the actual CSV data
                csv_lines = []
                for line in content.split("\n"):
                    # Skip empty lines and obvious text descriptions
                    if not line.strip() or line.strip().startswith("Here") or "generated" in line.lower():
                        continue
                    
                    # Look for CSV header or data rows
                    if "," in line and (
                        # Header row check
                        ("X" in line and "A" in line and "y" in line) or 
                        # Data row check - contains comma-separated values that could be parsed as numbers
                        all(part.strip().replace('.', '', 1).replace('-', '', 1).isdigit() 
                            for part in line.split(",") if part.strip())
                    ):
                        csv_lines.append(line)
                
                if not csv_lines:
                    # Try to extract from code blocks if direct extraction failed
                    if "```" in content:
                        code_blocks = content.split("```")
                        for block in code_blocks:
                            if "," in block and ("X" in block or "x" in block):
                                # Found a potential CSV block
                                block_lines = block.strip().split("\n")
                                # Remove any language identifier
                                if block_lines and (block_lines[0].lower() == "csv" or block_lines[0].strip() == ""):
                                    block_lines = block_lines[1:]
                                csv_lines = block_lines
                                break
                
                if not csv_lines:
                    raise ValueError("Could not extract CSV data from response")
                
                # Create a string of just the CSV content and parse it
                csv_content = "\n".join(csv_lines)
                
                # For debugging
                print(f"Extracted CSV content:\n{csv_content[:200]}...")
                
                try:
                    df = pd.read_csv(StringIO(csv_content))
                    
                    # Clean up column names - they may have leading/trailing spaces
                    df.columns = [col.strip() for col in df.columns]
                    
                    column_mapping = {}
                    for col in df.columns:
                        if col.lower() == 'x': column_mapping[col] = 'X'
                        elif col.lower() == 'a': column_mapping[col] = 'A'
                        elif col.lower() == 'y': column_mapping[col] = 'y'
                    
                    if column_mapping:
                        df = df.rename(columns=column_mapping)
                    
                    # Fall back to positional columns if we still don't have them
                    if not all(col in df.columns for col in ['X', 'A', 'y']):
                        if len(df.columns) >= 3:
                            df = df.iloc[:, :3]
                            df.columns = ['X', 'A', 'y']
                        else:
                            raise ValueError(f"Not enough columns in CSV: {df.columns}")
                    
                    # Keep only the required columns in the right order
                    df = df[['X', 'A', 'y']]
                    
                    # Convert to numeric and clean
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Drop NaN rows
                    original_len = len(df)
                    df = df.dropna()
                    if len(df) < original_len:
                        print(f"Dropped {original_len - len(df)} rows with invalid/NaN values")
                    
                    # Make sure X is in expected range
                    df['X'] = df['X'].clip(-1.5, 1.5)
                    
                    # Ensure A is binary
                    df['A'] = (df['A'] > 0.5).astype(int)
                    
                    # Check if we have any valid data
                    if len(df) == 0:
                        raise ValueError("No valid data rows after processing")
                    
                    # Ensure we have the right number of samples (or fewer)
                    if len(df) > batch_size:
                        df = df.iloc[:batch_size]
                    
                    print(f"Successfully extracted {len(df)} valid data points")
                    return df
                
                except Exception as e:
                    print(f"Error parsing CSV data: {e}")
                    print(f"CSV content that failed to parse: {csv_content[:500]}...")
                    raise
            
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
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--num_samples", type=int, default=30000, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for API calls")
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