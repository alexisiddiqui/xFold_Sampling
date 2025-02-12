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
    sampling_steps: list[int] = field(default_factory=lambda: [25])
    recycling_steps: list[int] = field(default_factory=lambda: [1])
    diffusion_samples: list[int] = field(
        default_factory=lambda: [100]
    )  # this is ur batch size - use wisely
    step_scale: list[float] = field(default_factory=lambda: [1.638])
    output_format: list[str] = field(default_factory=lambda: ["pdb"])
    num_workers: list[int] = field(default_factory=lambda: [2])
    use_msa_server: list[bool] = field(default_factory=lambda: [True])
    override: list[bool] = field(default_factory=lambda: [True])
    msa_server_url: list[str] = field(default_factory=lambda: ["https://api.colabfold.com"])
    msa_pairing_strategy: list[str] = field(default_factory=lambda: ["greedy"])
    write_full_pae: list[bool] = field(default_factory=lambda: [False])
    write_full_pde: list[bool] = field(default_factory=lambda: [False])
    max_paired_seqs: List[int] = field(default_factory=lambda: [4,8,16,32,64,128])

    use_previous_msa: list[str] = field(
        default_factory=lambda: ["/home/alexi/Documents/xFold_Sampling/boltz_sample/HOIP_dab3/msa"]
    )
    num_seeds: int = 20  # Number of seeds to run for each parameter combination

    def __post_init__(self):
        # Ensure that the number of seeds is a positive integer
        self.seeds = list(range(self.num_seeds))


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


def plot_confidence_scores(global_json_path: str, output_path: str, group_by_seed: bool = True):
    """
    Creates a subplot for each parameter set showing confidence scores distribution with shared axes.
    When group_by_seed is True, combines all seeds into a single distribution per parameter set.
    """
    with open(global_json_path, "r") as f:
        data = json.load(f)

    # Group data by parameter set and seed
    param_sets = {}
    for item in data:
        run = item["run"]
        # Extract parameter set and seed number
        param_parts = run.split("_")
        try:
            # Try to convert last part to int to verify it's a seed number
            int(param_parts[-1])
            param_set = "_".join(param_parts[:-1])
        except ValueError:
            param_set = run

        if param_set not in param_sets:
            param_sets[param_set] = []
        param_sets[param_set].append(item["confidence_score"])

    if not group_by_seed:
        # If not grouping by seed, keep separate by run
        runs = {}
        for item in data:
            run = item["run"]
            if run not in runs:
                runs[run] = []
            runs[run].append(item["confidence_score"])
        param_sets = runs

    # Create subplot for each parameter set
    n_param_sets = len(param_sets)
    fig, axes = plt.subplots(n_param_sets, 1, figsize=(12, 4 * n_param_sets), squeeze=False)

    # Find global min and max for consistent axes
    all_scores = [score for scores in param_sets.values() for score in scores]
    global_min = min(all_scores)
    global_max = max(all_scores)

    for i, (param_set, scores) in enumerate(param_sets.items()):
        ax = axes[i, 0]
        sns.histplot(scores, ax=ax, bins=20)
        ax.set_title(f"Confidence Score Distribution - {param_set}")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
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
    base_name: str,
    output_dirs: List[str],
    global_json_path: str,
    output_path: str,
    group_by_seed: bool = False,
):
    """
    Performs PCA on CA positions using XTC trajectories with highest confidence PDB as topology.
    Creates scatter plots for all runs combined and individual runs with shared axes.
    Optionally groups different seeds of the same parameter set together.

    Args:
        base_name: Base name for files
        output_dirs: List of output directories
        global_json_path: Path to global JSON file
        output_path: Where to save the plot
        group_by_seed: If True, groups different seeds with same parameters together
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

        # Extract parameter set name (everything before the seed number if grouping by seed)
        if group_by_seed:
            # Split by underscore and remove the last part (seed number)
            param_parts = run_name.split("_")
            try:
                # Try to convert last part to int to verify it's a seed number
                int(param_parts[-1])
                param_set = "_".join(param_parts[:-1])
            except ValueError:
                # If last part isn't a number, use full run_name
                param_set = run_name
        else:
            param_set = run_name

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

        # Initialize parameter set if not exists
        if param_set not in runs:
            runs[param_set] = {"coordinates": [], "scores": [], "seeds": []}

        # Store run data
        runs[param_set]["coordinates"].extend(run_coords)
        runs[param_set]["scores"].extend(run_scores.get(run_name, [None] * len(run_coords)))
        if group_by_seed:
            runs[param_set]["seeds"].extend([run_name] * len(run_coords))

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

    # Plot individual parameter sets
    start_idx = 0
    for idx, (param_set, run_data) in enumerate(runs.items(), start=1):
        n_frames = len(run_data["coordinates"])
        end_idx = start_idx + n_frames

        # Plot using the global PCA projection
        run_coords_2d = coords_all_2d[start_idx:end_idx]
        run_scores = run_data["scores"]

        if group_by_seed and "seeds" in run_data:
            # Create different markers/colors for different seeds
            unique_seeds = list(set(run_data["seeds"]))
            for seed in unique_seeds:
                seed_mask = np.array(run_data["seeds"]) == seed
                seed_coords = run_coords_2d[seed_mask]
                seed_scores = np.array(run_scores)[seed_mask]

                scatter = axes[idx].scatter(
                    seed_coords[:, 0],
                    seed_coords[:, 1],
                    c=seed_scores,
                    cmap="viridis",
                    s=50,
                    alpha=0.7,
                    vmin=score_min,
                    vmax=score_max,
                    label=seed,
                )
            # axes[idx].legend(title="Seeds", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
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
        axes[idx].set_title(f"PCA of CA Positions - {param_set}")
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)
        plt.colorbar(scatter, ax=axes[idx], label="Confidence Score")

        start_idx = end_idx

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def create_visualizations(
    base_name: str, output_dirs: List[str], parent_dir: str, group_by_seed: bool = True
):
    """
    Creates all visualizations.
    Args:
        base_name: Base name for files
        output_dirs: List of output directories
        parent_dir: Parent directory for output
        group_by_seed: If True, groups different seeds with same parameters together in plots
    """
    # First create global JSON
    global_json_path = combine_all_json_files(base_name, output_dirs, parent_dir)

    # Create confidence score plots
    confidence_plot_path = os.path.join(parent_dir, f"{base_name}_confidence_distributions.png")
    plot_confidence_scores(global_json_path, confidence_plot_path, group_by_seed)

    # Create PCA plot
    pca_plot_path = os.path.join(parent_dir, f"{base_name}_pca_analysis.png")
    plot_pca_analysis(base_name, output_dirs, global_json_path, pca_plot_path, group_by_seed)

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

    this_script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_script_dir)

    input_path = f"{this_script_dir}/HOIP/HOIP_apo697.fasta"
    previous_msa_dir = f"{this_script_dir}/HOIP"

    params.use_previous_msa = [previous_msa_dir]

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
        max_paired_seqs,
        seed,
        msa_path,
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
        params.max_paired_seqs,
        params.seeds,
        params.use_previous_msa,
    ):
        suffix = f"steps{sampling_steps}_recycle{recycling_steps}_diff{diffusion_samples}_scale{step_scale}_maxseqs{max_paired_seqs}"
        output_name = f"{base_name}_{suffix}_{seed}"
        output_dir = os.path.join(parent_dir, output_name)  # Changed to parent_dir
        os.makedirs(output_dir, exist_ok=True)
        output_dirs.append(output_dir)
        cache_dir = "/data/localhost/not-backed-up/hussain/.boltz"
        if msa_path is not None:
            use_msa_server = False

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
            "--max_msa_seqs",
            str(max_paired_seqs),
            f"--seed={seed}",
            "--previous_msa_dir",
            str(msa_path),
            f"--cache={cache_dir}",

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

    # compress entire directory to a tar.gz file
    shutil.make_archive(
        parent_dir, "gztar", os.path.dirname(parent_dir), os.path.basename(parent_dir)
    )
    print(f"Compressed directory to {parent_dir}.tar.gz")

    # if compressing was successful, remove the original directory
    shutil.rmtree(parent_dir)

    print("All processing completed successfully.")


if __name__ == "__main__":
    main()
