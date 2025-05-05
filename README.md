# Bachelor Thesis: Using LLMs to generate observational datasets

The python code and relevant datasets are in the folders, for example in synthetic_experiment:

synthetic_data_generation: Is our script for generating datasets
run_utils: Trains models, creates the confidence intervals
main: Runs the specific experiments and store the results under the code folder
plot: Plots the results found under the figs folder
openai_synthetic_data_generator: Generates the observational dataset from the given prompt

The real world experiment does not include a synthetic_data_generation file rather csv_data_handler which handle both the rct and observational dataset csv files.

Only for the Yuxin DGP folder the plot and results are store within the folder itself rather than in standard figs or code folder.