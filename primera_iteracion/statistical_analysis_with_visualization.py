
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from statsmodels.stats.multicomp import MultiComparison

# Define file paths (update these paths if necessary)
file_paths = [
    './calibration_results_simplified/metrics_dataset_15_simplified.json',
    './calibration_results_simplified/metrics_dataset_31_simplified.json',
    './calibration_results_simplified/metrics_dataset_37_simplified.json',
    './calibration_results_simplified/metrics_dataset_44_simplified.json'
]

# Define metrics to analyze
metrics = ['Accuracy', 'Brier Score', 'Log Loss', 'ECE']

# Load the JSON files and extract metrics
results = {metric: {} for metric in metrics}
methods = None

for file_path in file_paths:
    with open(file_path, 'r') as f:
        data = json.load(f)
        if not methods:
            methods = list(data.keys())  # Get methods from the first dataset
        for metric in metrics:
            for method, values in data.items():
                if method not in results[metric]:
                    results[metric][method] = []
                results[metric][method].append(values[metric])

# Perform Friedman test for each metric
friedman_results = {}
for metric in metrics:
    scores_matrix = [results[metric][method] for method in methods]
    friedman_results[metric] = friedmanchisquare(*scores_matrix)

# Save Friedman results to JSON
friedman_json = {}
for metric, result in friedman_results.items():
    friedman_json[metric] = {
        "Statistic": result.statistic,
        "p-value": result.pvalue
    }
output_dir = "./statistical_analysis_results"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "friedman_results.json"), "w") as f:
    json.dump(friedman_json, f, indent=4)

# Perform post-hoc analysis (Tukey HSD) and save results
tukey_results_json = {}
for metric in metrics:
    scores_matrix = [results[metric][method] for method in methods]
    scores_flattened = np.array(scores_matrix).T.flatten()
    methods_repeated = np.tile(methods, len(file_paths))

    # Organize into a DataFrame for MultiComparison
    data = pd.DataFrame({
        'Score': scores_flattened,
        'Method': methods_repeated
    })

    # Perform Tukey HSD test
    mc = MultiComparison(data['Score'], data['Method'])
    tukey_results = mc.tukeyhsd()
    
    # Save Tukey results as JSON
    tukey_summary = {
        f"{pair[0]} vs {pair[1]}": {
            "Mean Diff": round(pair[2], 4),
            "p-value": round(pair[4], 4),
            "Significant": bool(pair[4] < 0.05)
        }
        for pair in tukey_results._results_table.data[1:]
    }
    tukey_results_json[metric] = tukey_summary

    # Save JSON file for Tukey HSD
    tukey_file = os.path.join(output_dir, f"tukey_results_{metric.replace(' ', '_').lower()}.json")
    with open(tukey_file, "w") as f:
        json.dump(tukey_summary, f, indent=4)

# Generate boxplots for visualization
for metric in metrics:
    data_to_plot = [results[metric][method] for method in methods]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=methods, showmeans=True)
    plt.title(f"Comparison of {metric} Across Calibration Methods")
    plt.ylabel(metric)
    plt.xlabel("Calibration Methods")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"boxplot_{metric.replace(' ', '_').lower()}.png"))
    plt.close()

print(f"Statistical analysis results and visualizations saved to {output_dir}.")
