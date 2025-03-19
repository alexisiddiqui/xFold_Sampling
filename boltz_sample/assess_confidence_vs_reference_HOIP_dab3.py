import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Define experiment directories
experiment_dirs = [
    "HOIP_dab3/HOIP_dab3_20250227_025859_0.25dropout",
    "HOIP_dab3/HOIP_dab3_20250227_114339_0dropout",
    "HOIP_dab3/HOIP_dab3_20250227_192756_0.5dropout",
    "HOIP_dab3/HOIP_dab3_20250228_014114_0.9dropout",
]


def extract_maxseqs(run_name):
    """Extract maxseqs value from run name."""
    match = re.search(r"maxseqs(\d+)_", run_name)
    if match:
        return int(match.group(1))
    return None


# Modified plot_top_structures_by_w1 to analyze across maxseqs values
def plot_top_structures_by_maxseqs(combined_df, metric_columns, output_dir, n_structures=10):
    """
    Plot the top N structures by W1 distance for each maxseqs value.

    Args:
        combined_df: DataFrame with all experiments and their metrics
        metric_columns: List of metric columns to plot
        output_dir: Directory to save the plots
        n_structures: Number of top structures to plot for each maxseqs value
    """
    # Extract maxseqs from run names
    combined_df["maxseqs"] = combined_df["run"].apply(extract_maxseqs)

    # Group by maxseqs
    maxseqs_values = sorted([val for val in combined_df["maxseqs"].unique() if val is not None])

    if not maxseqs_values:
        print("No valid maxseqs values found for top structures plot")
        return

    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(
        len(metric_columns) + 1, 1, figsize=(14, (len(metric_columns) + 1) * 4), sharex=True
    )

    # Add space for the title
    fig.suptitle(
        f"Top {n_structures} Structures by W1 Distance Across MSA Parameters", fontsize=16, y=0.98
    )

    # Prepare color map for maxseqs values
    color_map = plt.cm.get_cmap("viridis", len(maxseqs_values))
    colors = {val: color_map(i) for i, val in enumerate(maxseqs_values)}

    # Plot W1 distance
    ax = axes[0]

    for i, maxseqs in enumerate(maxseqs_values):
        # Filter data for this maxseqs value
        maxseqs_df = combined_df[combined_df["maxseqs"] == maxseqs]

        if maxseqs_df.empty:
            continue

        # Sort by W1 distance and get top N structures
        top_df = maxseqs_df.sort_values("w1_distance").head(n_structures)

        if top_df.empty:
            continue

        # Plot W1 distances
        positions = np.arange(i * (n_structures + 2), i * (n_structures + 2) + len(top_df))
        ax.bar(
            positions,
            top_df["w1_distance"],
            color=colors[maxseqs],
            alpha=0.7,
            label=f"maxseqs={maxseqs}",
        )

        # Add dropout rate as text annotations
        for j, (_, row) in enumerate(top_df.iterrows()):
            if "dropout_rate" in row and not pd.isna(row["dropout_rate"]):
                ax.text(
                    positions[j],
                    row["w1_distance"] * 1.05,
                    f"d={row['dropout_rate']}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_ylabel("W1 Distance")
    ax.set_title("W1 Distance (lower is better)")
    ax.legend(title="MSA Parameter", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Plot each metric
    for i, metric in enumerate(metric_columns):
        ax = axes[i + 1]

        for maxseqs in maxseqs_values:
            # Filter data for this maxseqs value
            maxseqs_df = combined_df[combined_df["maxseqs"] == maxseqs]

            if maxseqs_df.empty:
                continue

            # Sort by W1 distance and get top N structures
            top_df = maxseqs_df.sort_values("w1_distance").head(n_structures)

            if top_df.empty or top_df[metric].isna().all():
                continue

            # Plot metric values
            positions = np.arange(
                maxseqs_values.index(maxseqs) * (n_structures + 2),
                maxseqs_values.index(maxseqs) * (n_structures + 2) + len(top_df),
            )
            ax.bar(positions, top_df[metric], color=colors[maxseqs], alpha=0.7)

        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)

    # Set x-axis labels for the bottom subplot
    ax = axes[-1]

    # Create tick positions and labels
    all_positions = []
    all_labels = []

    for i, maxseqs in enumerate(maxseqs_values):
        maxseqs_df = combined_df[combined_df["maxseqs"] == maxseqs]
        top_df = maxseqs_df.sort_values("w1_distance").head(n_structures)

        if not top_df.empty:
            positions = np.arange(i * (n_structures + 2), i * (n_structures + 2) + len(top_df))
            labels = [f"{m}" for m in top_df["model_id"]]

            all_positions.extend(positions)
            all_labels.extend(labels)

    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=90)

    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Make room for the legend

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "top_structures_by_maxseqs.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


# Modified plot_correlation_heatmap to analyze across maxseqs values
def plot_correlation_heatmap_by_maxseqs(correlation_dfs, experiment_names, output_dir):
    """Create separate correlation heatmaps for each maxseqs value."""
    # Extract maxseqs from experiment names and add to correlation dataframes
    for i, (corr_df, exp_name) in enumerate(zip(correlation_dfs, experiment_names)):
        maxseqs = extract_maxseqs(exp_name)
        if maxseqs is not None:
            corr_df["maxseqs"] = maxseqs
            corr_df["experiment"] = exp_name
        else:
            corr_df["maxseqs"] = None
            corr_df["experiment"] = exp_name

    # Combine all correlation data
    all_corr_df = pd.concat(correlation_dfs, ignore_index=True)

    # Group by maxseqs
    maxseqs_values = sorted([val for val in all_corr_df["maxseqs"].unique() if val is not None])

    if not maxseqs_values:
        print("No valid maxseqs values found for correlation heatmap")
        return

    # Create a single figure with multiple heatmaps
    fig, axes = plt.subplots(
        1,
        len(maxseqs_values),
        figsize=(5 * len(maxseqs_values), 8),
        gridspec_kw={"width_ratios": [1] * len(maxseqs_values)},
    )

    # If only one maxseqs value, axes won't be an array
    if len(maxseqs_values) == 1:
        axes = [axes]

    for i, maxseqs in enumerate(maxseqs_values):
        # Filter data for this maxseqs value
        maxseqs_df = all_corr_df[all_corr_df["maxseqs"] == maxseqs]

        if maxseqs_df.empty:
            axes[i].text(
                0.5,
                0.5,
                f"No data for maxseqs={maxseqs}",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            continue

        # Create pivot table for heatmap
        pivot_df = maxseqs_df.pivot(index="Metric", columns="experiment", values="Pearson_r")

        # Create significance mask
        sig_pivot = maxseqs_df.pivot(index="Metric", columns="experiment", values="Pearson_p")
        sig_mask = sig_pivot < 0.05

        # Custom annotation formatter for significant values
        def fmt(val, is_sig):
            if pd.isna(val):
                return ""
            formatted = f"{val:.2f}"
            return f"$\\bf{{{formatted}}}$" if is_sig else formatted

        # Create annotation array with appropriate formatting
        annot_array = np.empty_like(pivot_df, dtype=object)
        for row in range(pivot_df.shape[0]):
            for col in range(pivot_df.shape[1]):
                val = pivot_df.iloc[row, col]
                is_sig = sig_mask.iloc[row, col] if not pd.isna(sig_mask.iloc[row, col]) else False
                annot_array[row, col] = fmt(val, is_sig)

        # Plot heatmap
        sns.heatmap(
            pivot_df,
            annot=annot_array,
            cmap="coolwarm_r",  # Reversed so negative (good) correlations are in red
            vmin=-1,
            vmax=1,
            center=0,
            fmt="",
            ax=axes[i],
            cbar=i == len(maxseqs_values) - 1,  # Only show colorbar for last plot
        )

        axes[i].set_title(f"maxseqs = {maxseqs}")

        # Make experiment labels more readable
        if pivot_df.columns.size > 0:
            # Extract dropout from experiment names for cleaner labels
            dropout_labels = []
            for exp in pivot_df.columns:
                match = re.search(r"(\d+\.\d+|\d+)dropout", exp)
                if match:
                    dropout_labels.append(f"dropout={match.group(1)}")
                else:
                    dropout_labels.append(exp)

            axes[i].set_xticklabels(dropout_labels, rotation=45, ha="right")

    fig.suptitle(
        "Pearson Correlation with W1 Distance by MSA Parameter\n"
        "(Negative values indicate higher confidence → lower W1 distance)\n"
        "(Bold values are statistically significant p<0.05)",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "correlation_heatmap_by_maxseqs.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


# Modified plot_metrics_by_dropout to analyze across maxseqs values
def plot_metrics_by_maxseqs_and_dropout(combined_df, metric_columns, output_dir):
    """Create plots showing metrics vs dropout rate, separated by maxseqs value."""
    # Extract dropout rate and maxseqs
    combined_df["dropout_rate"] = combined_df["experiment"].apply(extract_dropout_rate)
    combined_df["maxseqs"] = combined_df["run"].apply(extract_maxseqs)

    # Filter out rows where dropout rate couldn't be extracted
    valid_df = combined_df.dropna(subset=["dropout_rate", "maxseqs"])

    if valid_df.empty:
        print("Warning: No valid dropout rates or maxseqs values. Skipping analysis.")
        return

    # Get unique values
    dropout_rates = sorted(valid_df["dropout_rate"].unique())
    maxseqs_values = sorted(valid_df["maxseqs"].unique())

    # For each metric (plus W1 distance), create a plot comparing maxseqs values across dropout rates
    for metric in metric_columns + ["w1_distance"]:
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

        # Create a color map for maxseqs values
        color_map = plt.cm.get_cmap("viridis", len(maxseqs_values))
        markers = [
            "o",
            "s",
            "D",
            "^",
            "v",
            "<",
            ">",
            "p",
            "*",
            "h",
        ]  # Different markers for clarity

        for i, maxseqs in enumerate(maxseqs_values):
            # Filter data for this maxseqs value
            maxseqs_df = valid_df[valid_df["maxseqs"] == maxseqs]

            if maxseqs_df.empty:
                continue

            # Calculate mean and std for each dropout rate
            agg_data = []
            for dropout in dropout_rates:
                dropout_df = maxseqs_df[maxseqs_df["dropout_rate"] == dropout]
                if not dropout_df.empty and not dropout_df[metric].isna().all():
                    agg_data.append(
                        {
                            "dropout_rate": dropout,
                            "mean": dropout_df[metric].mean(),
                            "std": dropout_df[metric].std(),
                            "count": dropout_df[metric].count(),
                        }
                    )

            if not agg_data:
                continue

            agg_df = pd.DataFrame(agg_data)

            # Plot with error bars
            ax.errorbar(
                agg_df["dropout_rate"],
                agg_df["mean"],
                yerr=agg_df["std"],
                marker=markers[i % len(markers)],
                markersize=8,
                linestyle="-",
                linewidth=2,
                capsize=5,
                label=f"maxseqs={maxseqs}",
                color=color_map(i),
                alpha=0.8,
            )

            # Add sample count as text
            for _, row in agg_df.iterrows():
                ax.annotate(
                    f"n={int(row['count'])}",
                    (row["dropout_rate"], row["mean"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    color=color_map(i),
                )

        # Add second plot showing metric vs maxseqs for each dropout rate
        ax2 = ax.inset_axes([0.65, 0.65, 0.3, 0.3])

        for i, dropout in enumerate(dropout_rates):
            # Filter data for this dropout rate
            dropout_df = valid_df[valid_df["dropout_rate"] == dropout]

            if dropout_df.empty:
                continue

            # Calculate mean and std for each maxseqs value
            agg_data = []
            for maxseqs in maxseqs_values:
                maxseqs_subset = dropout_df[dropout_df["maxseqs"] == maxseqs]
                if not maxseqs_subset.empty and not maxseqs_subset[metric].isna().all():
                    agg_data.append(
                        {
                            "maxseqs": maxseqs,
                            "mean": maxseqs_subset[metric].mean(),
                            "std": maxseqs_subset[metric].std(),
                        }
                    )

            if not agg_data:
                continue

            agg_df = pd.DataFrame(agg_data)

            # Plot with error bars in the inset
            ax2.errorbar(
                agg_df["maxseqs"],
                agg_df["mean"],
                yerr=agg_df["std"],
                marker="o",
                linestyle="-",
                capsize=3,
                label=f"dropout={dropout}",
                alpha=0.7,
            )

        # Format the main plot
        ax.set_xlabel("Dropout Rate")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Dropout Rate by MSA Parameter")
        ax.grid(True, alpha=0.3)
        ax.legend(title="MSA Parameter", loc="best")

        # Format the inset plot
        ax2.set_xscale("log")  # Log scale for maxseqs
        ax2.set_xlabel("maxseqs (log scale)")
        ax2.set_ylabel(metric)
        ax2.set_title("By Dropout Rate", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=6, title_fontsize=7)

        plt.tight_layout()

        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{metric}_by_maxseqs_and_dropout.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# Additional function to create boxplots for each metric by maxseqs value
def plot_boxplots_by_maxseqs(combined_df, metric_columns, output_dir):
    """Create boxplots for each metric, grouped by maxseqs value."""
    # Extract maxseqs
    combined_df["maxseqs"] = combined_df["run"].apply(extract_maxseqs)

    # Filter out rows where maxseqs couldn't be extracted
    valid_df = combined_df.dropna(subset=["maxseqs"])

    if valid_df.empty:
        print("Warning: No valid maxseqs values. Skipping boxplot analysis.")
        return

    # For each metric (plus W1 distance), create a boxplot grouped by maxseqs
    for metric in metric_columns + ["w1_distance"]:
        plt.figure(figsize=(12, 8))

        # Create boxplot
        ax = sns.boxplot(x="maxseqs", y=metric, data=valid_df, palette="viridis")

        # Add individual points
        sns.stripplot(x="maxseqs", y=metric, data=valid_df, color="black", alpha=0.4, size=4)

        # Add statistical annotations using scipy directly
        from scipy.stats import mannwhitneyu

        try:
            # Get unique maxseqs values
            maxseqs_values = sorted(valid_df["maxseqs"].unique())

            # Generate all pairs for comparison
            box_pairs = [
                (a, b) for i, a in enumerate(maxseqs_values) for b in maxseqs_values[i + 1 :]
            ]

            # Calculate statistics and add annotations
            y_range = valid_df[metric].max() - valid_df[metric].min()

            for i, (a, b) in enumerate(box_pairs):
                # Get data for the two groups
                group_a = valid_df[valid_df["maxseqs"] == a][metric].dropna()
                group_b = valid_df[valid_df["maxseqs"] == b][metric].dropna()

                # Skip if insufficient data
                if len(group_a) < 2 or len(group_b) < 2:
                    continue

                # Run Mann-Whitney U test
                stat, p = mannwhitneyu(group_a, group_b)

                # Add annotation for significant results
                if p < 0.05:
                    # Get x-coordinates
                    idx_a = maxseqs_values.index(a)
                    idx_b = maxseqs_values.index(b)

                    # Calculate y position for the bracket
                    y_pos = valid_df[metric].max() + (i + 1) * y_range * 0.05

                    # Add significance annotation
                    x = (idx_a + idx_b) / 2
                    sig_symbol = (
                        "*" if p < 0.05 else "**" if p < 0.01 else "***" if p < 0.001 else "ns"
                    )
                    ax.annotate(
                        f"{sig_symbol} p={p:.3f}",
                        xy=(x, y_pos),
                        xycoords="data",
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                    # Draw a line connecting the boxes
                    ax.plot([idx_a, idx_b], [y_pos, y_pos], "k-", linewidth=0.5)
                    ax.plot([idx_a, idx_a], [y_pos - y_range * 0.01, y_pos], "k-", linewidth=0.5)
                    ax.plot([idx_b, idx_b], [y_pos - y_range * 0.01, y_pos], "k-", linewidth=0.5)
        except Exception as e:
            print(f"Could not add statistical annotations to {metric} boxplot: {e}")

        plt.title(f"{metric} by MSA Parameter")
        plt.xlabel("MSA Parameter (maxseqs)")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{metric}_boxplot_by_maxseqs.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# Function to create a combined matrix plot showing the effect of both dropout and maxseqs
def plot_parameter_matrix(combined_df, metric_columns, output_dir):
    """Create a matrix plot showing the interaction between dropout rate and maxseqs for each metric."""
    # Extract parameters
    combined_df["dropout_rate"] = combined_df["experiment"].apply(extract_dropout_rate)
    combined_df["maxseqs"] = combined_df["run"].apply(extract_maxseqs)

    # Filter out rows where parameters couldn't be extracted
    valid_df = combined_df.dropna(subset=["dropout_rate", "maxseqs"])

    if valid_df.empty:
        print("Warning: No valid parameter values. Skipping matrix analysis.")
        return

    # Get unique values
    dropout_rates = sorted(valid_df["dropout_rate"].unique())
    maxseqs_values = sorted(valid_df["maxseqs"].unique())

    for metric in metric_columns + ["w1_distance"]:
        # Create a matrix of mean values
        matrix_data = []

        for dropout in dropout_rates:
            for maxseqs in maxseqs_values:
                subset = valid_df[
                    (valid_df["dropout_rate"] == dropout) & (valid_df["maxseqs"] == maxseqs)
                ]

                if not subset.empty and not subset[metric].isna().all():
                    matrix_data.append(
                        {
                            "dropout_rate": dropout,
                            "maxseqs": maxseqs,
                            "mean": subset[metric].mean(),
                            "std": subset[metric].std(),
                            "count": subset[metric].count(),
                        }
                    )

        if not matrix_data:
            print(f"No valid data for {metric} matrix plot")
            continue

        matrix_df = pd.DataFrame(matrix_data)

        # Pivot the data for the heatmap
        pivot_df = matrix_df.pivot(index="dropout_rate", columns="maxseqs", values="mean")

        # Create the heatmap
        plt.figure(figsize=(12, 8))

        # For W1 distance, lower is better (use coolwarm_r)
        # For other metrics, higher is better (use coolwarm)
        cmap = "coolwarm_r" if metric == "w1_distance" else "coolwarm"

        ax = sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={"label": f"Mean {metric}"},
        )

        # Add sample counts as text
        for i, dropout in enumerate(pivot_df.index):
            for j, maxseqs in enumerate(pivot_df.columns):
                count_df = matrix_df[
                    (matrix_df["dropout_rate"] == dropout) & (matrix_df["maxseqs"] == maxseqs)
                ]
                if not count_df.empty:
                    count = count_df["count"].values[0]
                    ax.text(
                        j + 0.5,
                        i + 0.85,
                        f"n={count}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

        plt.title(f"Effect of Dropout Rate and MSA Parameter on {metric}")
        plt.xlabel("MSA Parameter (maxseqs)")
        plt.ylabel("Dropout Rate")
        plt.tight_layout()

        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{metric}_parameter_matrix.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()


# Function to load global confidence data
def load_global_confidence(exp_dir):
    """Load the global confidence JSON file for the given experiment directory."""
    json_path = os.path.join(exp_dir, "HOIP_dab3_global_confidence.json")
    if not os.path.exists(json_path):
        print(f"Warning: Global confidence file not found at {json_path}")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)
    return data


# Function to extract confidence metrics
def extract_confidence_metrics(confidence_data):
    """Extract relevant confidence metrics from the confidence data."""
    metrics = []
    for item in confidence_data:
        # Extract model_id from pdb_name if not present
        model_id = item.get("model_id", None)
        pdb_name = item.get("pdb_name", "")

        if model_id is None and pdb_name:
            # Try to extract model_id from pdb_name (e.g., "HOIP_dab3_model_0")
            match = re.search(r"model_(\d+)", pdb_name)
            if match:
                model_id = match.group(0)  # "model_0"

        metric = {
            "run": item.get("run", ""),
            "confidence_score": item.get("confidence_score", None),
            "ptm": item.get("ptm", None),
            "iptm": item.get("iptm", None),
            "ligand_iptm": item.get("ligand_iptm", None),
            "protein_iptm": item.get("protein_iptm", None),
            "complex_plddt": item.get("complex_plddt", None),
            "complex_iplddt": item.get("complex_iplddt", None),
            "complex_pde": item.get("complex_pde", None),
            "complex_ipde": item.get("complex_ipde", None),
            "pdb_name": pdb_name,
            "model_id": model_id,
        }
        metrics.append(metric)
    return pd.DataFrame(metrics)


# Improved function to process a batch of PDB files for W1 distance calculation
def process_batch(batch_data):
    """
    Process a batch of PDB files and calculate W1 distances.
    This is a top-level function to enable multiprocessing.

    Args:
        batch_data: Tuple containing (batch_files, reference_pdb_path, alignment_info)
            - batch_files: List of (pdb_path, run, model_id) tuples
            - reference_pdb_path: Path to reference PDB file
            - alignment_info: Pre-computed alignment info or None

    Returns:
        List of dictionaries with W1 distance results
    """
    import os

    import MDAnalysis as mda
    from scipy.spatial.distance import pdist
    from scipy.stats import wasserstein_distance

    batch_files, reference_pdb_path, alignment_info = batch_data
    results = []

    # Load reference structure once for this batch
    try:
        ref_universe = mda.Universe(reference_pdb_path)
    except Exception as e:
        print(f"Error loading reference PDB in batch processing: {e}")
        return results

    # Get pre-calculated reference data if available
    ref_indices = None
    ref_distances = None
    if alignment_info is not None and "reference_distances" in alignment_info:
        ref_distances = alignment_info["reference_distances"]
        ref_indices = alignment_info["ref_indices"]

    for pdb_path, run, model_id in batch_files:
        try:
            # Load model PDB
            model_universe = mda.Universe(pdb_path)
            model_ca_atoms = model_universe.select_atoms("name CA")

            # If we have pre-computed alignment info, use it
            if alignment_info is not None and "model_indices" in alignment_info:
                # Use pre-calculated alignment
                model_indices = alignment_info["model_indices"]

                # Check that the number of model CA atoms matches what we expect
                if len(model_indices) > model_ca_atoms.n_atoms:
                    print(
                        f"Warning: Model {os.path.basename(pdb_path)} has fewer CA atoms than expected"
                    )
                    continue

                # Get aligned positions
                model_coords = model_ca_atoms.positions[model_indices]
                model_distances = pdist(model_coords)

                # Calculate W1 distance with reference
                w1_dist = wasserstein_distance(model_distances, ref_distances)
            else:
                # Calculate alignment for this specific model
                # First try to get reference CA atoms
                if ref_indices is None:
                    ref_ca_atoms = ref_universe.select_atoms("name CA")
                else:
                    ref_ca_atoms = ref_universe.select_atoms("name CA")

                # Improved alignment: Use chain-based alignment
                # Create a list of tuples with (chain, resid) for both reference and model
                ref_chain_residues = [(atom.segid, atom.resid) for atom in ref_ca_atoms]
                model_chain_residues = [(atom.segid, atom.resid) for atom in model_ca_atoms]

                # Find matching residues based on chain and sequential position
                ref_indices = []
                model_indices = []

                # Get unique chains in both structures
                ref_chains = set(cr[0] for cr in ref_chain_residues)
                model_chains = set(cr[0] for cr in model_chain_residues)

                # Match residues chain by chain
                for chain in ref_chains.intersection(model_chains):
                    # Get residues for this chain
                    ref_chain_resids = [
                        i for i, cr in enumerate(ref_chain_residues) if cr[0] == chain
                    ]
                    model_chain_resids = [
                        i for i, cr in enumerate(model_chain_residues) if cr[0] == chain
                    ]

                    # Match by sequential position in the chain (since numbering matches from start of each chain)
                    min_length = min(len(ref_chain_resids), len(model_chain_resids))
                    for i in range(min_length):
                        ref_indices.append(ref_chain_resids[i])
                        model_indices.append(model_chain_resids[i])

                # If no chain-based matches found, fall back to sequence position matching
                if not ref_indices:
                    min_length = min(len(ref_ca_atoms), len(model_ca_atoms))
                    ref_indices = list(range(min_length))
                    model_indices = list(range(min_length))

                # Extract coordinates for matched residues
                ref_coords = ref_ca_atoms.positions[ref_indices]
                if ref_distances is None:
                    ref_distances = pdist(ref_coords)

                model_coords = model_ca_atoms.positions[model_indices]
                model_distances = pdist(model_coords)

                # Calculate W1 distance
                w1_dist = wasserstein_distance(model_distances, ref_distances)

            results.append(
                {
                    "run": run,
                    "model_id": model_id,
                    "pdb_path": os.path.basename(pdb_path),
                    "w1_distance": w1_dist,
                    "n_aligned_residues": len(ref_indices),  # Add count of aligned residues
                }
            )

        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")

    return results


def compute_w1_distances(exp_dir, reference_pdb_path=None, batch_size=10, num_workers=4):
    """
    Compute W1 distances between each model PDB and the reference structure.
    Uses multiprocessing for efficiency with improved alignment.
    """
    import concurrent.futures

    import MDAnalysis as mda
    from scipy.spatial.distance import pdist
    from tqdm import tqdm

    if reference_pdb_path is None:
        # Look for the reference PDB in common locations
        reference_candidates = [
            os.path.join(os.path.dirname(exp_dir), "6sc6.pdb"),  # Same dir as experiment
            os.path.join(exp_dir, "6sc6.pdb"),  # Inside experiment dir
            "6sc6.pdb",  # Current directory
        ]

        for candidate in reference_candidates:
            if os.path.exists(candidate):
                reference_pdb_path = candidate
                break

        if reference_pdb_path is None:
            print("Error: Reference PDB not found. Please specify the path explicitly.")
            return None

    print(f"Using reference PDB: {reference_pdb_path}")

    # Load reference structure
    try:
        ref_universe = mda.Universe(reference_pdb_path)
    except Exception as e:
        print(f"Error loading reference PDB: {e}")
        return None

    # Find the highest confidence PDB to use as a representative model
    # This is used to establish proper residue alignment
    highest_conf_path = None
    for name in [
        f"{os.path.basename(exp_dir)}_highest_confidence.pdb",
        "HOIP_dab3_highest_confidence.pdb",
    ]:
        path = os.path.join(exp_dir, name)
        if os.path.exists(path):
            highest_conf_path = path
            break

    # Prepare alignment info
    alignment_info = None

    if highest_conf_path is not None:
        print(f"Using {highest_conf_path} for initial alignment")
        # Pre-calculate reference alignment using the highest confidence model
        try:
            model_universe = mda.Universe(highest_conf_path)
            ref_ca_atoms = ref_universe.select_atoms("name CA")
            model_ca_atoms = model_universe.select_atoms("name CA")

            print(f"Reference structure has {ref_ca_atoms.n_atoms} CA atoms")
            print(f"Model structure has {model_ca_atoms.n_atoms} CA atoms")

            # Improved alignment: Use chain-based alignment
            # Create a list of tuples with (chain, resid) for both reference and model
            ref_chain_residues = [(atom.segid, atom.resid) for atom in ref_ca_atoms]
            model_chain_residues = [(atom.segid, atom.resid) for atom in model_ca_atoms]

            # Get unique chains in both structures
            ref_chains = set(cr[0] for cr in ref_chain_residues)
            model_chains = set(cr[0] for cr in model_chain_residues)

            # Match residues chain by chain
            ref_indices = []
            model_indices = []

            # Match residues chain by chain
            for chain in ref_chains.intersection(model_chains):
                # Get residues for this chain
                ref_chain_resids = [i for i, cr in enumerate(ref_chain_residues) if cr[0] == chain]
                model_chain_resids = [
                    i for i, cr in enumerate(model_chain_residues) if cr[0] == chain
                ]

                # Match by sequential position in the chain (since numbering matches from start of each chain)
                min_length = min(len(ref_chain_resids), len(model_chain_resids))
                for i in range(min_length):
                    ref_indices.append(ref_chain_resids[i])
                    model_indices.append(model_chain_resids[i])

            # If no chain-based matches found, fall back to sequence position matching
            if not ref_indices:
                print("No chain-based matches found. Falling back to sequence position matching.")
                min_length = min(len(ref_ca_atoms), len(model_ca_atoms))
                ref_indices = list(range(min_length))
                model_indices = list(range(min_length))

            print(f"Aligned {len(ref_indices)} residues between reference and model")

            # Extract reference coordinates for common residues
            ref_coords = ref_ca_atoms.positions[ref_indices]
            reference_distances = pdist(ref_coords)

            # Store alignment info for reuse with all models
            alignment_info = {
                "ref_indices": ref_indices,
                "model_indices": model_indices,
                "reference_distances": reference_distances,
            }
        except Exception as e:
            print(f"Error establishing reference alignment: {e}")
            alignment_info = None

    # Find all PDB files in the experiment directory
    pdb_files = []
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file.endswith(".pdb") and "model_" in file:
                pdb_path = os.path.join(root, file)

                # Extract run from directory path
                run_path = os.path.dirname(pdb_path)
                while os.path.basename(run_path) != os.path.basename(exp_dir) and run_path != "/":
                    run_path = os.path.dirname(run_path)

                run = os.path.relpath(os.path.dirname(os.path.dirname(pdb_path)), run_path)
                if run == ".":
                    run = os.path.basename(exp_dir)

                # Extract model_id from filename
                model_id = None
                match = re.search(r"model_(\d+)", file)
                if match:
                    model_id = f"model_{match.group(1)}"

                pdb_files.append((pdb_path, run, model_id))

    if not pdb_files:
        print(f"No PDB files found in {exp_dir}")
        return None

    print(f"Found {len(pdb_files)} PDB files to process")

    # Process PDB files in batches using multiple workers
    w1_data = []
    batches = [pdb_files[i : i + batch_size] for i in range(0, len(pdb_files), batch_size)]

    # Prepare batch data for processing
    batch_data = [(batch, reference_pdb_path, alignment_info) for batch in batches]

    # Set up progress bar
    print(f"Processing {len(batches)} batches with {num_workers} workers...")

    # Use single-process approach if num_workers is 1 (better for debugging)
    if num_workers == 1:
        for i, data in enumerate(batch_data):
            print(f"Processing batch {i + 1}/{len(batch_data)}...")
            batch_results = process_batch(data)
            w1_data.extend(batch_results)

            # Periodically save progress
            if (i + 1) % 10 == 0 or i == len(batch_data) - 1:
                if w1_data:
                    temp_df = pd.DataFrame(w1_data)
                    temp_output_path = os.path.join(exp_dir, "HOIP_dab3_w1_distances_partial.csv")
                    temp_df.to_csv(temp_output_path, index=False)
                    print(f"Progress saved: {len(w1_data)} models processed so far")
    else:
        # Use multiprocessing for multiple workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches for processing
            future_to_batch = {
                executor.submit(process_batch, data): i for i, data in enumerate(batch_data)
            }

            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_batch), total=len(batches)
            ):
                batch_index = future_to_batch[future]
                try:
                    batch_results = future.result()
                    w1_data.extend(batch_results)

                    # Periodically save progress
                    if (batch_index + 1) % 10 == 0 or batch_index == len(batches) - 1:
                        if w1_data:
                            temp_df = pd.DataFrame(w1_data)
                            temp_output_path = os.path.join(
                                exp_dir, "HOIP_dab3_w1_distances_partial.csv"
                            )
                            temp_df.to_csv(temp_output_path, index=False)
                            print(f"Progress saved: {len(w1_data)} models processed so far")

                except Exception as e:
                    print(f"Error processing batch {batch_index}: {e}")

    # Create DataFrame and save to CSV
    if w1_data:
        w1_df = pd.DataFrame(w1_data)
        output_path = os.path.join(exp_dir, "HOIP_dab3_w1_distances.csv")
        w1_df.to_csv(output_path, index=False)
        print(f"W1 distances saved to {output_path}")
        return w1_df

    return None


def extract_w1_distances(
    exp_dir, reference_pdb_path=None, force_compute=False, batch_size=10, num_workers=4
):
    """Extract W1 distances from the saved data or compute them if not available."""
    # Look for existing W1 distance data (could be in JSON or in a plot)
    w1_data = []
    runs = []

    # If force_compute is True, skip searching for existing data and compute from scratch
    if force_compute:
        print(f"Force computing W1 distances for {exp_dir}")
        return compute_w1_distances(exp_dir, reference_pdb_path, batch_size, num_workers)

    # First check if there's a global W1 distances file
    w1_json_path = os.path.join(exp_dir, "HOIP_dab3_w1_distances.json")
    if os.path.exists(w1_json_path):
        with open(w1_json_path, "r") as f:
            data = json.load(f)

        # Convert to DataFrame
        w1_df = pd.DataFrame(
            [
                {
                    "run": item.get("run", ""),
                    "model_id": item.get("model_id", ""),
                    "w1_distance": item.get("w1_distance", None),
                }
                for item in data
            ]
        )
        return w1_df

    # Check for CSV file with W1 distances
    w1_csv_path = os.path.join(exp_dir, "HOIP_dab3_w1_distances.csv")
    if os.path.exists(w1_csv_path):
        return pd.read_csv(w1_csv_path)

    # If no W1 distances file exists, compute them
    print(f"No W1 distances found for {exp_dir}. Computing from PDB files...")
    return compute_w1_distances(exp_dir, reference_pdb_path, batch_size, num_workers)


# Improved function to merge confidence metrics with W1 distances
def merge_data(confidence_df, w1_df):
    """Merge confidence metrics with W1 distances."""
    if confidence_df is None or w1_df is None:
        return None

    # Clean up model_id in both dataframes to ensure consistent matching
    def clean_model_id(model_id):
        if pd.isna(model_id) or model_id is None:
            return None
        match = re.search(r"model_(\d+)", str(model_id))
        return f"model_{match.group(1)}" if match else model_id

    confidence_df["clean_model_id"] = confidence_df["model_id"].apply(clean_model_id)
    w1_df["clean_model_id"] = w1_df["model_id"].apply(clean_model_id)

    # First try to merge on run and clean_model_id
    merged_df = pd.merge(
        confidence_df, w1_df, on=["run", "clean_model_id"], how="inner", suffixes=("", "_w1")
    )

    # If that didn't work well, try alternative methods
    if len(merged_df) < min(len(confidence_df) * 0.1, 10):
        print(
            "Warning: Initial merge resulted in very few matches. Trying alternative merge strategies."
        )

        # Try just matching on clean_model_id if run may not match exactly
        merged_df = pd.merge(
            confidence_df, w1_df, on=["clean_model_id"], how="inner", suffixes=("", "_w1")
        )

        # If still insufficient matches, try matching just on model numbers
        if len(merged_df) < min(len(confidence_df) * 0.1, 10):
            print("Still insufficient matches. Extracting model numbers for matching...")

            # Extract model numbers from model_id
            def extract_model_number(model_id):
                if pd.isna(model_id) or model_id is None:
                    return None
                match = re.search(r"model_(\d+)", str(model_id))
                return int(match.group(1)) if match else None

            confidence_df["model_number"] = confidence_df["model_id"].apply(extract_model_number)
            w1_df["model_number"] = w1_df["model_id"].apply(extract_model_number)

            # Merge on model number
            merged_df = pd.merge(
                confidence_df, w1_df, on=["model_number"], how="inner", suffixes=("", "_w1")
            )

    # Print merge statistics
    print(
        f"Successfully merged {len(merged_df)} entries out of {len(confidence_df)} confidence scores and {len(w1_df)} W1 distances"
    )

    # Clean up the merged dataframe
    if "model_id_w1" in merged_df.columns:
        merged_df.drop("model_id_w1", axis=1, inplace=True)
    if "clean_model_id" in merged_df.columns:
        merged_df.drop("clean_model_id", axis=1, inplace=True)
    if "model_number" in merged_df.columns:
        merged_df.drop("model_number", axis=1, inplace=True)

    return merged_df


# Function to calculate correlations
def calculate_correlations(df, metric_columns):
    """Calculate Pearson and Spearman correlations between metrics and W1 distance."""
    correlations = {
        "Metric": [],
        "Pearson_r": [],
        "Pearson_p": [],
        "Spearman_r": [],
        "Spearman_p": [],
    }

    for metric in metric_columns:
        # Remove any NaN values for correlation calculation
        valid_data = df[[metric, "w1_distance"]].dropna()

        if len(valid_data) > 1:  # Need at least two data points for correlation
            pearson_r, pearson_p = pearsonr(valid_data[metric], valid_data["w1_distance"])
            spearman_r, spearman_p = spearmanr(valid_data[metric], valid_data["w1_distance"])

            correlations["Metric"].append(metric)
            correlations["Pearson_r"].append(pearson_r)
            correlations["Pearson_p"].append(pearson_p)
            correlations["Spearman_r"].append(spearman_r)
            correlations["Spearman_p"].append(spearman_p)
        else:
            # Add entry with NaN values if insufficient data
            correlations["Metric"].append(metric)
            correlations["Pearson_r"].append(np.nan)
            correlations["Pearson_p"].append(np.nan)
            correlations["Spearman_r"].append(np.nan)
            correlations["Spearman_p"].append(np.nan)

    return pd.DataFrame(correlations)


# Function to plot correlations
def plot_correlations(df, exp_name, metric_columns, output_dir):
    """Create scatter plots for each metric vs W1 distance."""
    for metric in metric_columns:
        # Skip if no valid data for this metric
        if df[metric].isna().all() or df["w1_distance"].isna().all():
            print(f"Skipping plot for {metric} due to insufficient data")
            continue

        plt.figure(figsize=(10, 6))

        # Create scatter plot with coloring by run
        runs = df["run"].unique()
        if len(runs) <= 10:  # Only color by run if there are a reasonable number
            for run in runs:
                run_data = df[df["run"] == run]
                valid_data = run_data[[metric, "w1_distance"]].dropna()
                if not valid_data.empty:
                    plt.scatter(
                        valid_data[metric],
                        valid_data["w1_distance"],
                        label=f"Run: {run}",
                        alpha=0.7,
                    )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            # If too many runs, use a single color
            valid_data = df[[metric, "w1_distance"]].dropna()
            plt.scatter(valid_data[metric], valid_data["w1_distance"], alpha=0.7)

        # Add regression line
        valid_data = df[[metric, "w1_distance"]].dropna()
        if len(valid_data) > 1:
            # Use numpy to calculate regression line
            x = valid_data[metric].values
            y = valid_data["w1_distance"].values

            m, b = np.polyfit(x, y, 1)
            plt.plot(x, m * x + b, "r-")

            # Calculate and display correlation
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)

            title = (
                f"{exp_name} - {metric} vs W1 Distance\n"
                f"Pearson r: {pearson_r:.3f} (p: {pearson_p:.3g})\n"
                f"Spearman r: {spearman_r:.3f} (p: {spearman_p:.3g})"
            )
        else:
            title = f"{exp_name} - {metric} vs W1 Distance\nInsufficient data for correlation"

        plt.title(title)
        plt.xlabel(metric)
        plt.ylabel("W1 Distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{metric}_vs_w1.png"))
        plt.close()


# Function to create a combined correlation heatmap
def plot_correlation_heatmap(correlation_dfs, experiment_names, output_dir):
    """Create a heatmap showing correlations for all experiments."""
    # Prepare data for heatmap
    heatmap_data = []

    for corr_df, exp_name in zip(correlation_dfs, experiment_names):
        for _, row in corr_df.iterrows():
            # Include all values, even if NaN
            heatmap_data.append(
                {
                    "Experiment": exp_name,
                    "Metric": row["Metric"],
                    "Pearson_r": row["Pearson_r"],
                    "Significant": row["Pearson_p"] < 0.05,
                }
            )

    if not heatmap_data:
        print("Warning: No valid correlation data for heatmap")
        return

    heatmap_df = pd.DataFrame(heatmap_data)

    # Create pivot table for heatmap
    pivot_df = heatmap_df.pivot(index="Metric", columns="Experiment", values="Pearson_r")

    # Create significance mask
    sig_pivot = heatmap_df.pivot(index="Metric", columns="Experiment", values="Significant")

    # Custom annotation formatter - use scientific notation and bold for significant values
    def fmt(val, is_sig):
        if pd.isna(val):
            return ""
        # Format as scientific notation with 2 decimal places
        formatted = f"{val:.2f}"  # Changed from scientific to fixed-point format for clarity
        # Bold if significant
        return f"$\\bf{{{formatted}}}$" if is_sig else formatted

    # Create annotation array with appropriate formatting
    annot_array = np.empty_like(pivot_df, dtype=object)
    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            val = pivot_df.iloc[i, j]
            is_sig = sig_pivot.iloc[i, j] if not pd.isna(sig_pivot.iloc[i, j]) else False
            annot_array[i, j] = fmt(val, is_sig)

    # Plot heatmap showing all correlations but with significant ones in bold
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_df,
        annot=annot_array,
        cmap="coolwarm_r",  # Reversed colormap so negative correlations (good) are in red
        vmin=-1,
        vmax=1,
        center=0,
        fmt="",  # Empty format as we're providing our own annotations
    )
    plt.title(
        "Pearson Correlation with W1 Distance\n(Negative values indicate higher confidence → lower W1 distance)\n(Bold values are statistically significant p<0.05)"
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


# Function to extract dropout rate from experiment name
def extract_dropout_rate(exp_name):
    """Extract dropout rate from experiment name."""
    match = re.search(r"(\d+\.\d+|\d+)dropout", exp_name)
    if match:
        return float(match.group(1))
    return np.nan


# Function to plot metrics by dropout rate
def plot_metrics_by_dropout(combined_df, metric_columns, output_dir):
    """Create violin plots of metrics by dropout rate."""
    # Extract dropout rate
    combined_df["dropout_rate"] = combined_df["experiment"].apply(extract_dropout_rate)

    # Filter out rows where dropout rate couldn't be extracted
    valid_df = combined_df.dropna(subset=["dropout_rate"])

    if valid_df.empty:
        print("Warning: No valid dropout rates extracted. Skipping dropout analysis.")
        return

    # Plot each metric vs dropout rate using violin plots
    for metric in metric_columns + ["w1_distance"]:
        # Skip if this metric is all NaN
        if valid_df[metric].isna().all():
            print(f"Skipping {metric} - no valid data")
            continue

        plt.figure(figsize=(12, 8))

        # Create violin plot
        ax = sns.violinplot(
            x="dropout_rate", y=metric, data=valid_df, palette="viridis", inner="quartile", cut=0
        )

        # Add individual data points
        sns.stripplot(
            x="dropout_rate", y=metric, data=valid_df, color="black", alpha=0.3, size=3, jitter=True
        )

        # Add count labels for each dropout rate
        for i, dropout in enumerate(sorted(valid_df["dropout_rate"].unique())):
            count = valid_df[valid_df["dropout_rate"] == dropout][metric].count()
            plt.annotate(
                f"n={count}",
                (i, valid_df[valid_df["dropout_rate"] == dropout][metric].max()),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.title(f"{metric} Distribution by Dropout Rate")
        plt.xlabel("Dropout Rate")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{metric}_vs_dropout_violin.png"))
        plt.close()


# Function to plot top structures by W1 distance
def plot_top_structures_by_w1(combined_df, metric_columns, output_dir, n_structures=20):
    """
    Plot the top N structures by W1 distance along with all confidence metrics.
    Points are colored by dropout rate.

    Args:
        combined_df: DataFrame with all experiments and their metrics
        metric_columns: List of metric columns to plot
        output_dir: Directory to save the plots
        n_structures: Number of top structures to plot (default: 20)
    """
    # Check if we have dropout information
    if "dropout_rate" not in combined_df.columns:
        combined_df["dropout_rate"] = combined_df["experiment"].apply(extract_dropout_rate)

    # Sort by W1 distance and get top N structures
    top_df = combined_df.sort_values("w1_distance").head(n_structures)

    if top_df.empty:
        print("No valid data for top structures plot")
        return

    # Create a figure with subplots for each metric + W1 distance
    n_metrics = len(metric_columns) + 1  # +1 for W1 distance itself
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, n_metrics * 3), sharex=True)

    # Plot W1 distance on the first axis
    ax = axes[0]
    scatter = ax.scatter(
        range(len(top_df)),
        top_df["w1_distance"],
        c=top_df["dropout_rate"],
        cmap="viridis",
        alpha=0.8,
        s=100,
    )
    ax.set_ylabel("W1 Distance")
    ax.set_title(f"Top {n_structures} Structures by W1 Distance")
    ax.grid(True, alpha=0.3)

    # Plot each metric on subsequent axes
    for i, metric in enumerate(metric_columns):
        ax = axes[i + 1]

        # Skip if metric is all NaN for these structures
        if top_df[metric].isna().all():
            ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
            continue

        scatter = ax.scatter(
            range(len(top_df)),
            top_df[metric],
            c=top_df["dropout_rate"],
            cmap="viridis",
            alpha=0.8,
            s=100,
        )
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    # Add a colorbar for dropout rate
    cbar = fig.colorbar(scatter, ax=axes, orientation="vertical", pad=0.01)
    cbar.set_label("Dropout Rate")

    # Annotate each point with model_id on the x-axis of the bottom subplot
    ax = axes[-1]
    ax.set_xticks(range(len(top_df)))
    ax.set_xticklabels(
        [f"{m}\n{r}" for m, r in zip(top_df["model_id"], top_df["run"])], rotation=90
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "top20_structures_by_w1.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


# Main function to run the analysis
def main(reference_pdb_path=None, force_compute_w1=True, batch_size=10, num_workers=4):
    """
    Run the analysis of confidence metrics vs W1 distances.

    Args:
        reference_pdb_path: Path to the reference PDB file (default: will search for 6sc6.pdb)
        force_compute_w1: Whether to force computation of W1 distances (default: False)
        batch_size: Number of PDB files to process in each batch for W1 calculation (default: 10)
        num_workers: Number of parallel workers for W1 calculation (default: 4)
    """
    # Set base directory (change this to your actual base directory)
    base_dir = "."  # Current directory
    output_dir = os.path.join(base_dir, "_analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    # Define metric columns to analyze
    metric_columns = [
        "confidence_score",
        "ptm",
        "iptm",
        "protein_iptm",
        "complex_plddt",
        "complex_iplddt",
        "complex_pde",
        "complex_ipde",
    ]

    # Process each experiment directory
    all_correlation_dfs = []
    all_data = []
    experiment_names = []

    for exp_dir in experiment_dirs:
        exp_path = os.path.join(base_dir, exp_dir)
        if not os.path.exists(exp_path):
            print(f"Warning: Experiment directory {exp_path} not found. Skipping.")
            continue

        print(f"\nProcessing {exp_dir}...")
        exp_name = os.path.basename(exp_dir)
        experiment_names.append(exp_name)

        # Load confidence data
        confidence_data = load_global_confidence(exp_path)
        if confidence_data is None:
            print(f"Skipping {exp_dir} due to missing confidence data.")
            continue

        # Extract confidence metrics
        confidence_df = extract_confidence_metrics(confidence_data)

        # Extract W1 distances or compute them if needed
        w1_df = extract_w1_distances(
            exp_path, reference_pdb_path, force_compute_w1, batch_size, num_workers
        )

        if w1_df is None or w1_df.empty:
            print(f"Skipping {exp_dir} due to missing W1 distance data.")
            continue

        # Merge data
        merged_df = merge_data(confidence_df, w1_df)
        if merged_df is None or merged_df.empty:
            print(f"Skipping {exp_dir} due to empty merged data.")
            continue

        # Add experiment name to the dataframe
        merged_df["experiment"] = exp_name
        all_data.append(merged_df)

        # Calculate correlations
        corr_df = calculate_correlations(merged_df, metric_columns)
        all_correlation_dfs.append(corr_df)

        # Plot correlations for this experiment
        plot_correlations(merged_df, exp_name, metric_columns, output_dir)

        # Save merged dataframe
        merged_df.to_csv(os.path.join(output_dir, f"{exp_name}_merged_data.csv"), index=False)

        # Print correlation summary
        print(f"\nCorrelation summary for {exp_name}:")
        print(corr_df[["Metric", "Pearson_r", "Pearson_p"]].to_string(index=False))
        print("\n")

    # Create a combined dataframe with all experiments
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, "all_experiments_data.csv"), index=False)

        # Add plot of top structures by W1 distance
        plot_top_structures_by_w1(combined_df, metric_columns, output_dir)

        # Create correlation heatmap
        plot_correlation_heatmap(all_correlation_dfs, experiment_names, output_dir)

        # Plot metrics by dropout rate
        plot_metrics_by_dropout(combined_df, metric_columns, output_dir)

        plot_top_structures_by_maxseqs(combined_df, metric_columns, output_dir)
        plot_correlation_heatmap_by_maxseqs(all_correlation_dfs, experiment_names, output_dir)
        plot_metrics_by_maxseqs_and_dropout(combined_df, metric_columns, output_dir)
        plot_boxplots_by_maxseqs(combined_df, metric_columns, output_dir)
        plot_parameter_matrix(combined_df, metric_columns, output_dir)

        # Create a summary table of all correlations
        all_correlations = pd.concat(
            [df.assign(Experiment=name) for df, name in zip(all_correlation_dfs, experiment_names)],
            ignore_index=True,
        )
        all_correlations.to_csv(os.path.join(output_dir, "all_correlations.csv"), index=False)

        print(f"Analysis complete. Results saved to {output_dir}")
    else:
        print("No valid data to analyze. Check experiment directories and file structure.")


# Run the main function with command-line arguments
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze correlation between confidence scores and W1 distances."
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        help="Path to reference PDB file (default: will search for 6sc6.pdb)",
    )
    parser.add_argument(
        "--force-compute",
        "-f",
        action="store_true",
        help="Force computation of W1 distances even if they exist",
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default=".",
        help="Base directory containing experiment folders (default: current directory)",
    )
    parser.add_argument(
        "--experiments",
        "-e",
        type=str,
        nargs="+",
        help="Specific experiment directories to analyze (default: use predefined list)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of PDB files to process in each batch for W1 calculation (default: 10)",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers for W1 calculation (default: 4)",
    )

    args = parser.parse_args()

    # Update experiment directories if specified
    if args.experiments:
        experiment_dirs.clear()
        experiment_dirs.extend(args.experiments)

    # Change working directory if specified
    if args.base_dir != ".":
        os.chdir(args.base_dir)

    # Run main function with parsed arguments
    main(
        reference_pdb_path=args.reference,
        # force_compute_w1=args.force_compute,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
