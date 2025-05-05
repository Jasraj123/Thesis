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

class ObservationalDataGenerator:
    def __init__(self, api_key=None, model="gpt-4o"):

        self.api_key = api_key or OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.output_dir = Path("generated_data")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_rct_data(self, rct_file):
        """Load RCT data and remove y column if present."""
        print(f"Loading RCT data from {rct_file}...")
        self.rct_data = pd.read_csv(rct_file)
        print(f"Loaded RCT data with shape {self.rct_data.shape}")
        
        if 'y' in self.rct_data.columns:
            self.features_only_data = self.rct_data.drop(columns=['y'])
            print("Removed y column from RCT data")
        else:
            self.features_only_data = self.rct_data.copy()
        
        print(f"Features data shape: {self.features_only_data.shape}")
        print("Column statistics:")
        print(self.features_only_data.describe())
        
        return self.features_only_data
    
    def generate_data(self, total_samples=30000, batch_size=1000):
        """
        Generate synthetic observational data:
        1. Generate y values for existing RCT features
        2. Generate additional samples to reach total_samples
        3. Combine into one dataset
        """
        output_file = self.output_dir / f"synthetic_stroke_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        num_rct_samples = len(self.features_only_data)
        num_additional_samples = total_samples - num_rct_samples
        
        print(f"Generating data with {num_rct_samples} RCT samples and {num_additional_samples} additional samples")
        
        #Generate y values for existing RCT samples
        rct_with_y = self._generate_y_for_rct(batch_size)
        
        #Generate additional synthetic samples
        additional_data = self._generate_additional_samples(num_additional_samples, batch_size)
        
        #Combine datasets
        final_data = pd.concat([rct_with_y, additional_data], ignore_index=True)
        
        # Save the final dataset
        final_data.to_csv(output_file, index=False)
        print(f"Final dataset with {len(final_data)} samples saved to {output_file}")
        
        return final_data
    
    def _generate_y_for_rct(self, batch_size):
        """Generate y values for existing RCT features using OpenAI."""
        print(f"Generating y values for {len(self.features_only_data)} RCT samples...")
        
        # Create a copy of the RCT data to add y values
        rct_with_y = self.features_only_data.copy()
        
        # Process in batches
        all_y_values = np.zeros(len(rct_with_y))
        num_batches = (len(rct_with_y) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(rct_with_y))
            
            print(f"Processing RCT batch {batch_idx+1}/{num_batches} (samples {start_idx}-{end_idx})...")
            
            # Get batch features
            batch_features = rct_with_y.iloc[start_idx:end_idx].reset_index(drop=True)
            
            prompt = """
            Generate realistic binary outcome (y) values for these patients in a stroke study, based on the design and findings of the International Stroke Trial (IST), a large randomized controlled trial conducted between 1991 and 1996.
            
            Background:
            - The IST evaluated early use of aspirin in patients with acute ischaemic stroke.
            - In this dataset, A=1 represents aspirin treatment, A=0 represents no aspirin.
            - Patient characteristics such as age and systolic blood pressure at randomization were associated with outcome.
            - ST1PE represents stroke type (a categorical variable). Stroke types may have different associated risks.
            
            Output requirements:
            - Do not insert any missing or null values.

            Features:
            - AGE: Patient's age in years
            - RSBP: Randomization systolic blood pressure
            - ST1PE: Stroke type (categorical, must not contain null values)
            - A: Aspirin treatment indicator (1 = received aspirin, 0 = no aspirin)
            - y: Outcome (1 = patient died, 0 = patient survived)
            
            Here are the patient features:
            {features}
            
            Return ONLY a CSV with columns AGE,RSBP,ST1PE,A,y where AGE,RSBP,ST1PE,A are kept exactly the same, 
            and y contains your generated outcomes (0 or 1) for each patient.
            """.format(features=batch_features.to_csv(index=False))
            
            # Generate y values
            y_values = self._call_openai_with_retry(prompt, batch_features)
            
            # Store in the full array
            all_y_values[start_idx:end_idx] = y_values
        
        # Add generated y values to the RCT data
        rct_with_y['y'] = all_y_values
        
        print(f"Successfully generated y values for all RCT samples")
        print(f"RCT mortality rate: {rct_with_y['y'].mean():.2%}")
        
        return rct_with_y
    
    def _generate_additional_samples(self, num_samples, batch_size):
        """Generate additional synthetic samples (AGE, RSBP, ST1PE, A, y) to reach desired total."""
        if num_samples <= 0:
            return pd.DataFrame(columns=['AGE', 'RSBP', 'ST1PE', 'A', 'y'])
            
        print(f"Generating {num_samples} additional synthetic samples...")
        
        # Process in batches
        additional_data_list = []
        remaining_samples = num_samples
        
        while remaining_samples > 0:
            batch_size_actual = min(batch_size, remaining_samples)
            print(f"Generating batch of {batch_size_actual} samples ({remaining_samples} remaining)...")
            
            prompt = """
            Generate {num_samples} new synthetic patient records for a stroke study, based on the design and findings of the International Stroke Trial (IST), a large randomized controlled trial conducted between 1991 and 1996.

            Reference data from the International Stroke Trial (use to guide data generation):
            {reference_data}

            Background:
            - The IST evaluated early use of aspirin in patients with acute ischaemic stroke.
            - In this dataset, A=1 represents aspirin treatment, A=0 represents no aspirin.
            - Patient characteristics such as age and systolic blood pressure at randomization were associated with outcome.
            - ST1PE represents stroke type (a categorical variable). Stroke types may have different associated risks.

            Features:
            - AGE: Patient's age in years
            - RSBP: Randomization systolic blood pressure
            - ST1PE: Stroke type (categorical, must not contain null values)
            - A: Aspirin treatment indicator (1 = received aspirin, 0 = no aspirin)
            - y: Outcome (1 = patient died, 0 = patient survived)

            Output requirements:
            - Learn from distributions and patterns in the reference data.
            - Do not insert any missing or null values.

            Return ONLY a valid CSV with {num_samples} rows and columns AGE,RSBP,ST1PE,A,y containing your synthetic data.""".format(
                num_samples=batch_size_actual,
                reference_data=self.features_only_data.sample(min(20, len(self.features_only_data))).to_csv(index=False)
            )
            
            # Generate batch of synthetic data
            response = self._call_openai_for_synthetic_data(prompt, batch_size_actual)
            additional_data_list.append(response)
            remaining_samples -= len(response)
        
        # Combine all batches
        additional_data = pd.concat(additional_data_list, ignore_index=True)
        
        print(f"Successfully generated {len(additional_data)} additional synthetic samples")
        print(f"Additional data mortality rate: {additional_data['y'].mean():.2%}")
        
        return additional_data
    
    def _call_openai_with_retry(self, prompt, batch_features, max_retries=3):
        """Call OpenAI API to generate y values with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates realistic medical outcome data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                
                # Extract content from response
                content = response.choices[0].message.content
                
                cleaned_content = content
                if "```" in content:
                    parts = content.split("```")
                    for part in parts:
                        if "AGE" in part and "RSBP" in part and "ST1PE" in part and "A" in part:
                            cleaned_content = part.strip()
                            if cleaned_content.startswith("csv") or cleaned_content.startswith("CSV"):
                                cleaned_content = cleaned_content[3:].strip()
                            break
                
                # Save raw response for debugging
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_file = self.output_dir / f"raw_response_{timestamp}.txt"
                with open(raw_file, "w") as f:
                    f.write(cleaned_content)
                print(f"Raw response saved to {raw_file}")
                
                if "```" in content:
                    parts = content.split("```")
                    for part in parts:
                        if "AGE" in part and "RSBP" in part and "ST1PE" in part and "A" in part:
                            content = part.strip()
                            if content.startswith("csv") or content.startswith("CSV"):
                                content = content[3:].strip()
                            break
                
                df = pd.read_csv(StringIO(content))
                
                if 'y' in df.columns:
                    y_values = df['y'].values
                elif 'Y' in df.columns:
                    y_values = df['Y'].values
                else:
                    raise ValueError("No y column found in response")
                
                if len(y_values) < len(batch_features):
                    print(f"Warning: Generated {len(y_values)} values but needed {len(batch_features)}")
                    print(f"Processing in multiple parts...")
                    
                    # Keep track of the values we already have
                    all_values = np.zeros(len(batch_features))
                    all_values[:len(y_values)] = y_values
                    
                    remaining_start = len(y_values)
                    chunk_size = 200  
                    
                    while remaining_start < len(batch_features):
                        remaining_end = min(remaining_start + chunk_size, len(batch_features))
                        remaining_chunk = batch_features.iloc[remaining_start:remaining_end]
                        
                        chunk_prompt ="""
                        Generate realistic binary outcome (y) values for these patients in a stroke study, based on the design and findings of the International Stroke Trial (IST), a large randomized controlled trial conducted between 1991 and 1996.
                        
                        Background:
                        - The IST evaluated early use of aspirin in patients with acute ischaemic stroke.
                        - In this dataset, A=1 represents aspirin treatment, A=0 represents no aspirin.
                        - Patient characteristics such as age and systolic blood pressure at randomization were associated with outcome.
                        - ST1PE represents stroke type (a categorical variable). Stroke types may have different associated risks.
                        
                        Output requirements:
                        - Do not insert any missing or null values.

                        Features:
                        - AGE: Patient's age in years
                        - RSBP: Randomization systolic blood pressure
                        - ST1PE: Stroke type (categorical, must not contain null values)
                        - A: Aspirin treatment indicator (1 = received aspirin, 0 = no aspirin)
                        - y: Outcome (1 = patient died, 0 = patient survived)
                        
                        Here are the patient features:
                        {features}
                        
                        Return ONLY a CSV with columns AGE,RSBP,ST1PE,A,y where AGE,RSBP,ST1PE,A are kept exactly the same, 
                        and y contains your generated outcomes (0 or 1) for each patient.
                        """.format(features=batch_features.to_csv(index=False))
                        
                        print(f"Generating values for chunk {remaining_start}-{remaining_end}...")
                        
                        chunk_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that generates realistic medical outcome data."},
                                {"role": "user", "content": chunk_prompt}
                            ],
                            temperature=0.2,
                            max_tokens=2000
                        )
                        
                        
                    try: 
                        if "```" in content:
                            parts = content.split("```")
                            for part in parts:
                                if "AGE" in part and "RSBP" in part and "ST1PE" in part and "A" in part:
                                    content = part.strip()
                                    if content.startswith("csv") or content.startswith("CSV"):
                                        content = content[3:].strip()
                                    break
                        
                        df = pd.read_csv(StringIO(content))
                        
                        # Extract y values, handling different column names
                        if 'y' in df.columns:
                            y_values = df['y'].values
                        elif 'Y' in df.columns:
                            y_values = df['Y'].values
                        else:
                            raise ValueError("No y column found in response")
                            
                        chunk_size_actual = min(len(chunk_y_values), remaining_end - remaining_start)
                        all_values[remaining_start:remaining_start+chunk_size_actual] = chunk_y_values[:chunk_size_actual]
                            
                            # Move to next chunk
                        remaining_start += chunk_size_actual
                            
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                            # Continue to next chunk
                        remaining_start = remaining_end
                    
                    y_values = all_values
                
                # Ensure binary values
                if not all(value in [0, 1] for value in y_values):
                    raise ValueError("Generated values are not all binary (0 or 1)")
                
                y_values = np.array(y_values).astype(int)
                return y_values
            
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("All retry attempts failed.")
                    raise RuntimeError(f"Failed to generate y values after {max_retries} attempts: {e}")
    
    def _call_openai_for_synthetic_data(self, prompt, batch_size, max_retries=3):
        """Call OpenAI API to generate complete synthetic data with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates realistic medical outcome data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                
                # Extract content from response
                content = response.choices[0].message.content
                
                cleaned_content = content
                if "```" in content:
                    parts = content.split("```")
                    for part in parts:
                        if "AGE" in part and "RSBP" in part and "ST1PE" in part:
                            cleaned_content = part.strip()
                            if cleaned_content.startswith("csv") or cleaned_content.startswith("CSV"):
                                cleaned_content = cleaned_content[3:].strip()
                            break
                
                # Save raw response for debugging
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_file = self.output_dir / f"raw_synth_response_{timestamp}.txt"
                with open(raw_file, "w") as f:
                    f.write(cleaned_content)
                print(f"Raw synthetic response saved to {raw_file}")
                
                if "```" in content:
                    parts = content.split("```")
                    for part in parts:
                        if "," in part and "AGE" in part and "RSBP" in part and "ST1PE" in part:
                            content = part.strip()
                            if content.startswith("csv") or content.startswith("CSV"):
                                content = content[3:].strip()
                            break
                
                # Parse the CSV content
                df = pd.read_csv(StringIO(content))
                
                # Standardize column names
                column_mapping = {}
                for col in df.columns:
                    if col.lower() == 'age': column_mapping[col] = 'AGE'
                    elif col.lower() == 'rsbp': column_mapping[col] = 'RSBP'
                    elif col.lower() == 'st1pe': column_mapping[col] = 'ST1PE'
                    elif col.lower() == 'a': column_mapping[col] = 'A'
                    elif col.lower() == 'y': column_mapping[col] = 'y'
                    
                if column_mapping:
                    df = df.rename(columns=column_mapping)
                
                # Ensure we have all required columns
                required_cols = ['AGE', 'RSBP', 'ST1PE', 'A', 'y']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns. Got: {df.columns.tolist()}")
                
                df = df[required_cols]
                df = df.apply(pd.to_numeric, errors='coerce')
                
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic observational data for stroke study")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--rct_file", type=str, default="llm_training.csv", help="Reference RCT data file")
    parser.add_argument("--num_samples", type=int, default=30000, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size for API calls")
    
    args = parser.parse_args()
    
    generator = ObservationalDataGenerator(api_key=args.api_key, model=args.model)
    
    generator.load_rct_data(args.rct_file)
    
    synthetic_data = generator.generate_data(total_samples=args.num_samples, batch_size=args.batch_size) 