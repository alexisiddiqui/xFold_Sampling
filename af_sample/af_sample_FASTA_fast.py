import json
import os
import subprocess
from multiprocessing import Pool
from typing import Tuple

from tqdm import tqdm


def run_alphafold_command(args: Tuple[str, str, Tuple[int, int], str, int, int, bool]) -> dict:
    """
    Run a single AlphaFold command with the given parameters

    Args:
        args: Tuple containing (fasta_path, output_dir, max_MSA, fasta_header, num_seeds, num_recycle, use_dropout)

    Returns:
        dict: Dictionary containing the results for this run
    """
    fasta_path, output_dir, max_MSA, fasta_header, num_seeds, num_recycle, use_dropout = args

    print(f"Running alphafold for {fasta_header} with max MSA size {max_MSA}")
    os.makedirs(output_dir, exist_ok=True)

    max_MSA_string = f"{max_MSA[0]}:{max_MSA[1]}"
    dropout = "--use-dropout" if use_dropout else ""

    alphafold_command = [
        "colabfold_batch",
        fasta_path,
        output_dir,
        "--num-seeds",
        str(num_seeds),
        "--max-msa",
        max_MSA_string,
        dropout,
        "--num-recycle",
        str(num_recycle),
    ]

    subprocess.run(alphafold_command)

    # Process JSON results
    file_list = os.listdir(output_dir)
    json_list = [file for file in file_list if file.endswith(".json") and "rank" in file]

    results = {}
    results[max_MSA_string] = {}

    for idx, json_file in enumerate(json_list):
        with open(os.path.join(output_dir, json_file), "r") as file:
            json_data = json.load(file)
            json_data = {key: json_data[key] for key in ["plddt", "max_pae", "ptm"]}
            json_data["plddt"] = sum(json_data["plddt"]) / len(json_data["plddt"])
            results[max_MSA_string][idx] = json_data

    return results


def af_sample_FASTA_quick(
    fasta_path: str,
    output_dir: str = None,
    num_seeds: int = 1,
    max_seq_range: tuple[int, int] = (32, 256),
    max_models: int = 10000,
    use_dropout: bool = True,
    num_recycle: int = 1,
    num_threads: int = 5,  # Maximum number of MIG threads
):
    """
    Run AlphaFold sampling with multiprocessing support and progress bar

    Args:
        fasta_path: Path to the FASTA file
        output_dir: Output directory (optional)
        num_seeds: Number of seeds to use
        max_seq_range: Range of sequence lengths to sample
        max_models: Maximum number of models to generate
        use_dropout: Whether to use dropout
        num_recycle: Number of recycling iterations
        num_threads: Number of parallel threads to use (max 5 for MIG)
    """
    num_af_models = 5
    max_iter = max_models // (num_seeds * num_af_models)
    print(f"max_iter: {max_iter}")
    max_seq_stride = (max_seq_range[1] - max_seq_range[0]) // max_iter
    print(f"max_seq_stride: {max_seq_stride}")

    max_MSAs = [
        (max_seq // 2, max_seq)
        for max_seq in range(max_seq_range[0], max_seq_range[1], max_seq_stride)
    ]
    num_steps = len(max_MSAs)

    if output_dir is None:
        output_dir = (
            fasta_path.split(".")[0] + f"_{num_recycle}_af_sample_{num_steps}_{str(max_models)}"
        )

    # Read FASTA file
    with open(fasta_path, "r") as file:
        fasta_file = file.readlines()

    fasta_header = (
        fasta_file[0][1:].split()[0]
        if fasta_file[0][0] != ">"
        else fasta_path.split("/")[-1].split(".")[0]
    )

    print(
        f"Sampling {num_seeds} seeds for {fasta_header} over range {max_seq_range} with stride {max_seq_stride}"
    )

    output_dirs = [
        os.path.join(output_dir, f"maxMSA_{max_MSA[0]}_{max_MSA[1]}") for max_MSA in max_MSAs
    ]

    # Prepare arguments for multiprocessing
    args_list = [
        (fasta_path, out_dir, max_MSA, fasta_header, num_seeds, num_recycle, use_dropout)
        for max_MSA, out_dir in zip(max_MSAs, output_dirs)
    ]

    # Run processes with progress bar
    json_info = {}
    with Pool(processes=min(num_threads, 5)) as pool:
        results = list(
            tqdm(
                pool.imap(run_alphafold_command, args_list),
                total=len(args_list),
                desc="Processing AlphaFold runs",
            )
        )

        # Combine results
        for result in results:
            json_info.update(result)

    # Save the combined results
    json_name = output_dir + "_ranks.json"
    with open(json_name, "w") as file:
        json.dump(json_info, file)


if __name__ == "__main__":
    # Test the function
    fasta_path = "HOIP_dab3/HOIP_dab3.fasta"

    # change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    af_sample_FASTA_quick(
        fasta_path,
        num_recycle=3,
        num_seeds=4,
        max_models=101,
        num_threads=4,
    )

    print("Done")
