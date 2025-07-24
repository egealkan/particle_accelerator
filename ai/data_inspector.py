# ai/data_inspector.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading data for inspection...")
try:
    df = pd.read_csv("data/final_ai_data.csv")
except FileNotFoundError:
    print("❌ Error: Featured data file not found. Please run ai/data_preparer.py first.")
    exit()

# Separate the DataFrame into signal and background events
signal_df = df[df['label'] == 1]
background_df = df[df['label'] == 0]

print(f"Found {len(signal_df)} signal events and {len(background_df)} background events.")

# Define the features we want to inspect
features_to_inspect = [
    'muon_pt', 
    'antimuon_pt', 
    'delta_eta', 
    'delta_phi', 
    'delta_R'
]

# Create a plot for each feature
for feature in features_to_inspect:
    plt.figure(figsize=(10, 7))
    
    # Determine a common range for the histogram bins
    min_val = min(signal_df[feature].min(), background_df[feature].min())
    max_val = max(signal_df[feature].max(), background_df[feature].max())
    bins = np.linspace(min_val, max_val, 50)
    
    # Plot the histograms
    # We use density=True to normalize the histograms, as there are
    # different numbers of signal and background events.
    plt.hist(background_df[feature], bins=bins, alpha=0.5, density=True, label='Background')
    plt.hist(signal_df[feature], bins=bins, alpha=0.5, density=True, label='Signal')
    
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    
    # Save the plot
    plot_path = f"visualization/plots/inspection_{feature}.png"
    plt.savefig(plot_path)
    print(f"Saved inspection plot to {plot_path}")
    plt.close() # Close the plot to save memory