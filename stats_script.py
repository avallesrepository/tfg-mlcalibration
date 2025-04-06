import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# Define file paths (update these paths if necessary)
file_paths = [
    './calibration_results_xgboost_s/metrics_dataset_3_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_15_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_29_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_31_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_37_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_38_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_44_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_316_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_953_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_958_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_962_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_978_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_1067_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_1462_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_1464_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_1487_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_1494_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_1510_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_4134_repeated.json',
    './calibration_results_xgboost_s/metrics_dataset_40701_repeated.json'
]

# Define metrics to analyze
metrics = ['Accuracy']  #'Accuracy', 'Brier Score', 'Log Loss', 'ECE'

# Load the JSON files and extract metrics
results = {metric: {} for metric in metrics}
methods = None
datasets = []  # To track dataset IDs for the final CSV

for file_path in file_paths:
    dataset_id = os.path.basename(file_path).split('_')[2]  # Extract dataset ID from filename
    datasets.append(dataset_id)
    with open(file_path, 'r') as f:
        data = json.load(f)
        if not methods:
            methods = list(data.keys())  # Get methods from the first dataset
        for metric in metrics:
            for method, values in data.items():
                if method not in results[metric]:
                    results[metric][method] = []
                results[metric][method].append(values[metric])

# Create a DataFrame to store all metrics for each dataset
all_metrics_df = pd.DataFrame()
all_metrics_df['Dataset'] = datasets

for metric in metrics:
    for method in methods:
        column_name = f"{metric}_{method}"  # e.g., "Log Loss_Method A"
        all_metrics_df[column_name] = results[metric][method]

# Save the DataFrame to a CSV file
output_dir = "./accuracy/xgboost_s"
os.makedirs(output_dir, exist_ok=True)
all_metrics_file = os.path.join(output_dir, "all_metrics.csv")
all_metrics_df.to_csv(all_metrics_file, index=False)

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
with open(os.path.join(output_dir, "friedman_results.json"), "w") as f:
    json.dump(friedman_json, f, indent=4)

# Perform pairwise comparisons (Wilcoxon) and Benjamini-Hochberg adjustment
wilcoxon_results_json = {}
for metric in metrics:
    wilcoxon_results = {}
    scores_matrix = [results[metric][method] for method in methods]
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i >= j:
                continue
            # Perform Wilcoxon signed-rank test for paired data
            stat, p_value = wilcoxon(results[metric][method1], results[metric][method2])
            wilcoxon_results[f"{method1} vs {method2}"] = {
                "Statistic": stat,
                "p-value": p_value,
                "Significant": 'True' if p_value < 0.05 else 'False'  # Convert bool to string
            }
    
    # Apply Benjamini-Hochberg correction for multiple comparisons
    p_values = [result['p-value'] for result in wilcoxon_results.values()]
    rejected, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Update results with adjusted p-values
    for idx, (pair, result) in enumerate(wilcoxon_results.items()):
        result["Adjusted p-value"] = p_adjusted[idx]
        result["Significant"] = 'True' if p_adjusted[idx] < 0.05 else 'False'  # Convert bool to string
    
    wilcoxon_results_json[metric] = wilcoxon_results

    # Save Wilcoxon and Benjamini-Hochberg results
    wilcoxon_file = os.path.join(output_dir, f"wilcoxon_results_{metric.replace(' ', '_').lower()}.json")
    with open(wilcoxon_file, "w") as f:
        json.dump(wilcoxon_results, f, indent=4)

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
