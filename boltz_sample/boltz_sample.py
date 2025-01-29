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
    sampling_steps: list[int] = field(default_factory=lambda: [25, 200])
    recycling_steps: list[int] = field(default_factory=lambda: [10, 3])
    diffusion_samples: list[int] = field(default_factory=lambda: [100])
    step_scale: list[float] = field(default_factory=lambda: [1, 1.68, 2])
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
    Creates a subplot for each run showing confidence scores distribution.
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

    for i, (run, scores) in enumerate(runs.items()):
        ax = axes[i, 0]
        sns.histplot(scores, ax=ax, bins=20)
        ax.set_title(f"Confidence Score Distribution - {run}")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")

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
    Performs PCA on CA positions and creates a scatter plot colored by confidence scores.
    Additionally, plots each sub-run on its own axis within the same figure.
    """
    with open(global_json_path, "r") as f:
        data = json.load(f)

    # Group data by run
    runs = {}
    for item in data:
        run = item["run"]
        if run not in runs:
            runs[run] = {"coordinates": [], "scores": []}
        runs[run]["coordinates"].append(item["ca_coordinates"])
        runs[run]["scores"].append(item["confidence_score"])

    # Perform PCA on all data
    all_coordinates = np.array([item["ca_coordinates"] for item in data])
    all_scores = np.array([item["confidence_score"] for item in data])

    pca_all = PCA(n_components=2)
    coords_all_2d = pca_all.fit_transform(all_coordinates)

    fig, axes = plt.subplots(len(runs) + 1, 1, figsize=(10, 5 * (len(runs) + 1)))

    # Plot all runs in the first subplot
    ax_all = axes[0]
    scatter_all = ax_all.scatter(
        coords_all_2d[:, 0],
        coords_all_2d[:, 1],
        c=all_scores,
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    ax_all.set_xlabel(f"PC1 ({pca_all.explained_variance_ratio_[0]:.2%} variance)")
    ax_all.set_ylabel(f"PC2 ({pca_all.explained_variance_ratio_[1]:.2%} variance)")
    ax_all.set_title("PCA of CA Positions Colored by Confidence Score - All Runs")
    plt.colorbar(scatter_all, ax=ax_all, label="Confidence Score")

    # Plot each run in separate subplots
    for idx, (run, values) in enumerate(runs.items(), start=1):
        coordinates = np.array(values["coordinates"])
        scores = np.array(values["scores"])

        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(coordinates)

        scatter = axes[idx].scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=scores,
            cmap="viridis",
            s=50,
            alpha=0.7,
        )
        axes[idx].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        axes[idx].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        axes[idx].set_title(f"PCA of CA Positions - {run}")
        plt.colorbar(scatter, ax=axes[idx], label="Confidence Score")

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


# def find_highest_confidence_structure(base_name: str, output_dirs: List[str], final_pdb_path: str):
#     max_confidence = -1.0
#     best_pdb = ""
#     best_model_info = None

#     for output_dir in output_dirs:
#         confidence_json_path = os.path.join(
#             output_dir,
#             f"boltz_results_{base_name}",
#             "predictions",
#             f"{base_name}_combined_confidence.json",
#         )

#         if not os.path.exists(confidence_json_path):
#             print(f"Confidence JSON not found at {confidence_json_path}. Skipping.")
#             continue

#         try:
#             with open(confidence_json_path, "r") as jf:
#                 confidence_data = json.load(jf)

#                 for model in confidence_data:
#                     confidence_score = model.get("confidence_score", 0)
#                     if confidence_score > max_confidence:
#                         # Extract model number from pdb_name
#                         pdb_name = model.get("pdb_name", "")
#                         if pdb_name.startswith("confidence_"):
#                             model_num = pdb_name.split("model_")[-1]

#                             # Construct PDB path
#                             pdb_filename = f"{base_name}_model_{model_num}.pdb"
#                             potential_pdb = os.path.join(
#                                 output_dir,
#                                 f"boltz_results_{base_name}",
#                                 "predictions",
#                                 base_name,
#                                 pdb_filename,
#                             )

#                             if os.path.exists(potential_pdb):
#                                 max_confidence = confidence_score
#                                 best_pdb = potential_pdb
#                                 best_model_info = {
#                                     "model": model_num,
#                                     "confidence": confidence_score,
#                                     "ptm": model.get("ptm", "N/A"),
#                                     "plddt": model.get("complex_plddt", "N/A"),
#                                 }
#                                 print(
#                                     f"New best model found: {pdb_filename} with confidence {confidence_score:.4f}"
#                                 )

#         except json.JSONDecodeError as e:
#             print(f"Error reading JSON file {confidence_json_path}: {e}")
#             continue
#         except Exception as e:
#             print(f"Unexpected error processing {confidence_json_path}: {e}")
#             continue

#     if best_pdb and os.path.exists(best_pdb):
#         shutil.copy(best_pdb, final_pdb_path)
#         print("\nHighest confidence structure details:")
#         print(f"- Model: {best_model_info['model']}")
#         print(f"- Confidence Score: {best_model_info['confidence']:.4f}")
#         print(f"- pTM Score: {best_model_info['ptm']:.4f}")
#         print(f"- pLDDT Score: {best_model_info['plddt']:.4f}")
#         print(f"- Source: {best_pdb}")
#         print(f"- Copied to: {final_pdb_path}")
#     else:
#         print("\nNo valid PDB found for the highest confidence structure.")
#         if best_model_info:
#             print("Best model was identified but file could not be copied:")
#             print(f"- Expected path: {best_pdb}")
#             print(f"- Confidence score: {best_model_info['confidence']:.4f}")


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
