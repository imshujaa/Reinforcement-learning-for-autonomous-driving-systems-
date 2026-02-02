import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- CONFIGURATION ---
# List of agent techniques you want to compare
TECHNIQUES = ['RS', 'Q', 'CRL', 'DQN', 'CRL_DQN']

# Set the environment you want to plot
ENV_LABEL = "highway" # Change to "intersection" to plot the other environment

# The base path where your results are stored
BASE_PATH = f"output/DATA/{ENV_LABEL}/_test_pretrained/"
# ---------------------

# --- MAIN SCRIPT ---
all_agent_data = []

print("--- Loading Data ---")
# Loop through each technique to find its results file
for tech in TECHNIQUES:
    folder_path = os.path.join(BASE_PATH, tech)
    
    # Find the 'overall_results_....csv' file in the folder
    search_pattern = os.path.join(folder_path, "overall_results_*.csv")
    result_files = glob.glob(search_pattern)
    
    if not result_files:
        print(f"Warning: No results CSV found for technique '{tech}' in '{folder_path}'")
        continue

    # Load the most recent CSV file for that technique
    latest_file = max(result_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    
    # Add a column to label which agent this data belongs to
    df['agent'] = tech
    all_agent_data.append(df)
    print(f"Loaded data for '{tech}' from {os.path.basename(latest_file)}")

if not all_agent_data:
    print("\nError: No data was loaded. Please check your folder structure and file paths.")
else:
    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_agent_data, ignore_index=True)

    print("\n--- Generating Boxplot ---")
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=combined_df, x='agent', y='fail_rate', order=TECHNIQUES)
    
    # Add titles and labels
    plt.title(f'Agent Performance Comparison on {ENV_LABEL.capitalize()} Environment', fontsize=16)
    plt.ylabel('Failure Rate', fontsize=12)
    plt.xlabel('Agent Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot to a file
    output_dir = "output/PLOTS/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"agent_comparison_boxplot_{ENV_LABEL}.png")
    
    plt.savefig(plot_path)
    
    print(f"Boxplot successfully saved to: {plot_path}")
    
    # Display the plot on screen
    plt.show()