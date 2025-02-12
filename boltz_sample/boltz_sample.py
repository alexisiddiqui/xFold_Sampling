import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import List

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns
from MDAnalysis.coordinates.XTC import XTCWriter
from sklearn.decomposition import PCA

# Add at the top with other imports
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class BoltzParams:
    sampling_steps: list[int] = field(default_factory=lambda: [200])
    recycling_steps: list[int] = field(
        default_factory=lambda: [
            1,
        ]
    )
    diffusion_samples: list[int] = field(default_factory=lambda: [100])
    step_scale: list[float] = field(default_factory=lambda: [1])
    output_format: list[str] = field(default_factory=lambda: ["pdb"])
    num_workers: list[int] = field(default_factory=lambda: [4])
    override: list[bool] = field(default_factory=lambda: [True])
    use_msa_server: list[bool] = field(default_factory=lambda: [True])
    msa_server_url: list[str] = field(default_factory=lambda: ["https://api.colabfold.com"])
    msa_pairing_strategy: list[str] = field(default_factory=lambda: ["greedy"])
    write_full_pae: list[bool] = field(default_factory=lambda: [False])
    write_full_pde: list[bool] = field(default_factory=lambda: [False])


def combine_pdbs_to_xtc(predictions_dir: str, output_xtc: str):
    pdb_files = sorted(
        [
            os.path.join(predictions_dir, f)
            for f in os.listdir(predictions_dir)
            if f.endswith(".pdb")
        ]
    )

    if not pdb_files:
        print(f"No PDB files found in {predictions_dir}. Skipping XTC generation.")
        return

    print(f"Combining {len(pdb_files)} PDB files into {output_xtc}")

    # Load first PDB to initialize the universe
    u = mda.Universe(pdb_files[0])
    with XTCWriter(output_xtc, n_atoms=u.atoms.n_atoms) as xtc:
        xtc.write(u)
        for pdb in pdb_files[1:]:
            u = mda.Universe(pdb)
            xtc.write(u)
    print(f"XTC file created at {output_xtc}")


def combine_jsons(predictions_dir: str, output_json: str):
    json_files = sorted(
        [
            os.path.join(predictions_dir, f)
            for f in os.listdir(predictions_dir)
            if f.endswith(".json")
        ]
    )

    if not json_files:
        print(f"No JSON files found in {predictions_dir}. Skipping JSON combination.")
        return

    combined_data: List[dict] = []
    run_name = os.path.basename(os.path.dirname(os.path.dirname(predictions_dir)))
    for json_file in json_files:
        pdb_name = os.path.splitext(os.path.basename(json_file))[0]
        with open(json_file, "r") as jf:
            data = json.load(jf)
            if isinstance(data, dict):
                data["pdb_name"] = pdb_name
                data["run_name"] = run_name
                combined_data.append(data)
            else:
                print(f"Skipping non-dictionary entry in {json_file}: {data}")

    with open(output_json, "w") as out_jf:
        json.dump(combined_data, out_jf, indent=4)
    print(f"Combined JSON file created at {output_json}")


def combine_all_xtc(xtc_files: List[str], output_xtc: str):
    if not xtc_files:
        print("No XTC files to combine.")
        return

    print(f"Combining {len(xtc_files)} XTC files into {output_xtc}")

    first_xtc = xtc_files[0]
    u = mda.Universe(first_xtc)
    with XTCWriter(output_xtc, n_atoms=u.atoms.n_atoms) as xtc_writer:
        for xtc_file in xtc_files:
            print(f"Adding frames from {xtc_file}")
            u_xtc = mda.Universe(first_xtc)  # Reinitialize for each file
            u_xtc.load_new(xtc_file)
            for ts in u_xtc.trajectory:
                xtc_writer.write(u_xtc)
    print(f"All XTC files combined into {output_xtc}")


def combine_all_json_files(base_name: str, output_dirs: List[str], parent_dir: str) -> str:
    """
    Combines all confidence JSON files from different runs into a single global JSON file.
    Returns the path to the combined JSON file.
    """
    all_data = []

    for output_dir in output_dirs:
        run_name = os.path.basename(output_dir)
        json_path = os.path.join(
            output_dir,
            f"boltz_results_{base_name}",
            "predictions",
            f"{base_name}_combined_confidence.json",
        )

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                for item in data:
                    item["run"] = run_name
                all_data.extend(data)

    output_path = os.path.join(parent_dir, f"{base_name}_global_confidence.json")
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=4)

    return output_path


def plot_confidence_scores(global_json_path: str, output_path: str):
    """
    Creates a subplot for each run showing confidence scores distribution with shared axes.
    """
    with open(global_json_path, "r") as f:
        data = json.load(f)

    # Group data by run
    runs = {}
    for item in data:
        run = item["run"]
        if run not in runs:
            runs[run] = []
        runs[run].append(item["confidence_score"])

    # Create subplot for each run
    n_runs = len(runs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(10, 4 * n_runs), squeeze=False)

    # Find global min and max for consistent axes
    all_scores = [score for scores in runs.values() for score in scores]
    global_min = min(all_scores)
    global_max = max(all_scores)

    for i, (run, scores) in enumerate(runs.items()):
        ax = axes[i, 0]
        sns.histplot(scores, ax=ax, bins=20)
        ax.set_title(f"Confidence Score Distribution - {run}")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        # Set consistent x-axis limits
        ax.set_xlim(global_min, global_max)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_ca_coordinates(pdb_path: str) -> np.ndarray:
    """
    Extracts CA coordinates from a PDB file.
    """
    u = mda.Universe(pdb_path)
    ca_atoms = u.select_atoms("name CA")
    coords = ca_atoms.positions

    # Calculate pairwise coordinate differences
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    return diff


def plot_pca_analysis(
    base_name: str, output_dirs: List[str], global_json_path: str, output_path: str
):
    """
    Performs PCA on CA positions using XTC trajectories with highest confidence PDB as topology.
    Creates scatter plots for all runs combined and individual runs with shared axes.
    """
    # Load confidence scores from global JSON
    with open(global_json_path, "r") as f:
        data = json.load(f)

    # Create dictionary mapping run names to confidence scores
    run_scores = {}
    for item in data:
        run = item["run"]
        if run not in run_scores:
            run_scores[run] = []
        run_scores[run].append(item["confidence_score"])

    # Get the topology (highest confidence PDB)
    topology_pdb = os.path.join(
        os.path.dirname(global_json_path), f"{base_name}_highest_confidence.pdb"
    )
    if not os.path.exists(topology_pdb):
        raise FileNotFoundError(f"Topology PDB not found at {topology_pdb}")

    # Initialize universe with topology
    all_coordinates = []
    all_scores = []
    runs = {}

    # Process each run's XTC file
    for output_dir in output_dirs:
        run_name = os.path.basename(output_dir)
        xtc_path = os.path.join(
            output_dir, f"boltz_results_{base_name}", "predictions", f"{base_name}_combined.xtc"
        )

        if not os.path.exists(xtc_path):
            print(f"Warning: XTC file not found at {xtc_path}")
            continue

        # Load trajectory
        u = mda.Universe(topology_pdb, xtc_path)
        ca_atoms = u.select_atoms("name CA")

        # Extract coordinates and store them
        run_coords = []
        for ts in u.trajectory:
            coords = ca_atoms.positions.flatten()  # Flatten for PCA
            run_coords.append(coords)
            all_coordinates.append(coords)

        # Store run data
        runs[run_name] = {
            "coordinates": run_coords,
            "scores": run_scores.get(run_name, [None] * len(run_coords)),
        }
        all_scores.extend(run_scores.get(run_name, [None] * len(run_coords)))

    # Convert to numpy arrays
    all_coordinates = np.array(all_coordinates)
    all_scores = np.array(all_scores)

    # Perform PCA on all data
    pca_all = PCA(n_components=2)
    coords_all_2d = pca_all.fit_transform(all_coordinates)

    # Create figure with subplots
    fig, axes = plt.subplots(len(runs) + 1, 1, figsize=(10, 5 * (len(runs) + 1)))

    # Find global min and max for consistent axes and color scaling
    x_min, x_max = coords_all_2d[:, 0].min(), coords_all_2d[:, 0].max()
    y_min, y_max = coords_all_2d[:, 1].min(), coords_all_2d[:, 1].max()
    score_min, score_max = np.nanmin(all_scores), np.nanmax(all_scores)

    # Plot all runs combined
    scatter_all = axes[0].scatter(
        coords_all_2d[:, 0],
        coords_all_2d[:, 1],
        c=all_scores,
        cmap="viridis",
        s=50,
        alpha=0.7,
        vmin=score_min,
        vmax=score_max,
    )
    axes[0].set_xlabel(f"PC1 ({pca_all.explained_variance_ratio_[0]:.2%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca_all.explained_variance_ratio_[1]:.2%} variance)")
    axes[0].set_title("PCA of CA Positions - All Runs Combined")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    plt.colorbar(scatter_all, ax=axes[0], label="Confidence Score")

    # Plot individual runs
    start_idx = 0
    for idx, (run_name, run_data) in enumerate(runs.items(), start=1):
        n_frames = len(run_data["coordinates"])
        end_idx = start_idx + n_frames

        # Plot using the global PCA projection
        run_coords_2d = coords_all_2d[start_idx:end_idx]
        run_scores = run_data["scores"]

        scatter = axes[idx].scatter(
            run_coords_2d[:, 0],
            run_coords_2d[:, 1],
            c=run_scores,
            cmap="viridis",
            s=50,
            alpha=0.7,
            vmin=score_min,
            vmax=score_max,
        )
        axes[idx].set_xlabel(f"PC1 ({pca_all.explained_variance_ratio_[0]:.2%} variance)")
        axes[idx].set_ylabel(f"PC2 ({pca_all.explained_variance_ratio_[1]:.2%} variance)")
        axes[idx].set_title(f"PCA of CA Positions - {run_name}")
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)
        plt.colorbar(scatter, ax=axes[idx], label="Confidence Score")

        start_idx = end_idx

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_visualizations(base_name: str, output_dirs: List[str], parent_dir: str):
    """
    Creates all visualizations.
    """
    # First create global JSON
    global_json_path = combine_all_json_files(base_name, output_dirs, parent_dir)

    # Create confidence score plots
    confidence_plot_path = os.path.join(parent_dir, f"{base_name}_confidence_distributions.png")
    plot_confidence_scores(global_json_path, confidence_plot_path)

    # Create PCA plot
    pca_plot_path = os.path.join(parent_dir, f"{base_name}_pca_analysis.png")
    plot_pca_analysis(base_name, output_dirs, global_json_path, pca_plot_path)

    print("Visualizations created:")
    print(f"- Global JSON: {global_json_path}")
    print(f"- Confidence distributions: {confidence_plot_path}")
    print(f"- PCA analysis: {pca_plot_path}")


def find_highest_confidence_structure(base_name: str, output_dirs: List[str], final_pdb_path: str):
    max_confidence = -1.0
    best_pdb = ""

    for output_dir in output_dirs:
        confidence_json_path = os.path.join(
            output_dir,
            "boltz_results_" + base_name,
            "predictions",
            f"{base_name}_combined_confidence.json",
        )
        predictions_dir = os.path.join(
            output_dir, "boltz_results_" + base_name, "predictions", base_name
        )

        if not os.path.exists(confidence_json_path):
            print(f"Confidence JSON not found at {confidence_json_path}. Skipping.")
            continue

        with open(confidence_json_path, "r") as jf:
            confidence_data = json.load(jf)

            for model in confidence_data:
                confidence_score = model.get("confidence_score", 0)
                if confidence_score > max_confidence:
                    max_confidence = confidence_score
                    model_id = model.get("model_id", None)
                    if model_id is None:
                        # Infer model_id from pdb_name
                        pdb_name = model.get("pdb_name", "")
                        if not pdb_name:
                            continue
                        # Split pdb_name to extract model_id (e.g., "confidence_HOIP_apo697_model_0" -> 0)
                        parts = pdb_name.split("_")
                        model_id = parts[-1]  # Assumes the last part is the model number
                    if model_id is not None:
                        pdb_filename = f"{base_name}_model_{model_id}.pdb"
                        best_pdb = os.path.join(predictions_dir, pdb_filename)
                        print(
                            f"New best model found: {pdb_filename} with confidence {max_confidence}"
                        )

    if best_pdb and os.path.exists(best_pdb):
        shutil.copy(best_pdb, final_pdb_path)
        print(f"Highest confidence PDB ({max_confidence}) copied to {final_pdb_path}")
    else:
        print("No valid PDB found for the highest confidence structure.")


def main():
    params = BoltzParams()

    input_path = "/home/alexi/Documents/xFold_Sampling/boltz_sample/HOIP/HOIP_apo697.fasta"
    input_dir = os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    parent_dir = os.path.join(
        os.path.dirname(input_path), f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(parent_dir, exist_ok=True)
    output_dirs = []  # To keep track of all output directories

    for (
        sampling_steps,
        recycling_steps,
        diffusion_samples,
        step_scale,
        output_format,
        num_workers,
        override,
        use_msa_server,
        msa_server_url,
        msa_pairing_strategy,
        write_full_pae,
        write_full_pde,
    ) in product(
        params.sampling_steps,
        params.recycling_steps,
        params.diffusion_samples,
        params.step_scale,
        params.output_format,
        params.num_workers,
        params.override,
        params.use_msa_server,
        params.msa_server_url,
        params.msa_pairing_strategy,
        params.write_full_pae,
        params.write_full_pde,
    ):
        suffix = f"steps{sampling_steps}_recycle{recycling_steps}_diff{diffusion_samples}_scale{step_scale}"
        output_name = f"{base_name}_{suffix}"
        output_dir = os.path.join(parent_dir, output_name)  # Changed to parent_dir
        os.makedirs(output_dir, exist_ok=True)
        output_dirs.append(output_dir)

        cmd = [
            "boltz",
            "predict",
            input_path,
            "--use_msa_server" if use_msa_server else "",
            "--msa_server_url",
            msa_server_url,
            "--recycling_steps",
            str(recycling_steps),
            "--msa_pairing_strategy",
            msa_pairing_strategy,
            f"--sampling_steps={sampling_steps}",
            f"--diffusion_samples={diffusion_samples}",
            f"--step_scale={step_scale}",
            f"--output_format={output_format}",
            f"--num_workers={num_workers}",
            "--override" if override else "",
            "--write_full_pae" if write_full_pae else "",
            "--write_full_pde" if write_full_pde else "",
            "--out_dir",
            output_dir,
        ]

        # Remove empty strings from the command
        cmd = [arg for arg in cmd if arg]

        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)

        # Paths to predictions and combined files
        predictions_dir = os.path.join(
            output_dir, "boltz_results_" + base_name, "predictions", base_name
        )
        combined_xtc = os.path.join(
            output_dir, "boltz_results_" + base_name, "predictions", f"{base_name}_combined.xtc"
        )
        combined_json = os.path.join(
            output_dir,
            "boltz_results_" + base_name,
            "predictions",
            f"{base_name}_combined_confidence.json",
        )

        # Combine PDBs into XTC
        combine_pdbs_to_xtc(predictions_dir, combined_xtc)

        # Combine JSONs into a single JSON
        combine_jsons(predictions_dir, combined_json)

    # After all runs, combine all XTC files into a single XTC
    all_combined_xtc = [
        os.path.join(
            output_dir, "boltz_results_" + base_name, "predictions", f"{base_name}_combined.xtc"
        )
        for output_dir in output_dirs
    ]
    final_combined_xtc = os.path.join(parent_dir, f"{base_name}_all_combined.xtc")
    combine_all_xtc(all_combined_xtc, final_combined_xtc)

    # Find the highest confidence structure across all runs
    final_best_pdb = os.path.join(parent_dir, f"{base_name}_highest_confidence.pdb")
    find_highest_confidence_structure(base_name, output_dirs, final_best_pdb)

    combine_all_json_files(base_name, output_dirs, parent_dir)
    create_visualizations(base_name, output_dirs, parent_dir)

    print("All processing completed successfully.")


if __name__ == "__main__":
    main()
