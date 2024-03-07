# function to run alphafold using af sampling
# must be run in colabfold environment
import os
import subprocess

def af_sample_FASTA_quick(fasta_path:str,
                          output_dir:str=None,
                          num_seeds:int=20,
                          max_seq_range: tuple[int,int] = (2,256),
                          max_models = 1000, 
                          use_dropout:bool = True,
                          num_recycle:int = 1):

    num_af_models = 5
    if output_dir is None:
        output_dir = fasta_path.split(".")[0] + f"_af_sample_{str(max_models)}"

    # read in fasta or a3m file
    with open(fasta_path, "r") as file:
        fasta_file = file.readlines()

    if fasta_file[0][0] != ">":
        fasta_header = fasta_file[0][1:].split()[0]
    else:
        fasta_header = fasta_path.split("/")[-1].split(".")[0]

    # calculate the stride for the sequence sampling - this is the number of iterations
    # max_models = num_seeds * num_af_models * max_seq_stride
    max_iter = max_models // (num_seeds * num_af_models)
    print(f"max_iter: {max_iter}")
    max_seq_stride = (max_seq_range[1] - max_seq_range[0]) // max_iter
    print(f"max_seq_stride: {max_seq_stride}")
    print(f"Sampling {num_seeds} seeds for {fasta_header} over range {max_seq_range} with stride {max_seq_stride}")

    max_MSAs = [(max_seq//2, max_seq) for max_seq in range(max_seq_range[0], max_seq_range[1], max_seq_stride)]

    output_dirs = [os.path.join(output_dir, f"maxMSA_{max_MSA[0]}_{max_MSA[1]}") for max_MSA in max_MSAs]

    for max_MSA, output_dir in zip(max_MSAs, output_dirs):
        print(f"Running alphafold for {fasta_header} with max MSA size {max_MSA}")
        os.makedirs(output_dir, exist_ok=True)
        
        max_MSA_string = f"{max_MSA[0]}:{max_MSA[1]}"

        dropout = "--use-dropout" if use_dropout else ""

        alphafold_command = ["colabfold_batch",
                             fasta_path,
                             output_dir,
                             "--num-seeds", str(num_seeds),
                             "--max-msa", max_MSA_string,
                             dropout,
                             "--num-recycle", str(num_recycle)]
        
        print(f"Running command: {' '.join(alphafold_command)}")
        subprocess.run(alphafold_command)


if __name__ == "__main__":
    # test the function
    fasta_path  = "BPTI/P00974_60.fasta"
    fasta_path1 = "MBP/MBP_wt.fasta"
    fasta_path2 = "LXRa/LXRa.fasta"
    fasta_path3 = "HOIP/HOIP_apo697.fasta"
    fasta_path4 = "BRD4/BRD4_APO_484.fasta"
    # output_dir = fasta_path.split(".")[0] + "_af_sample"

    # af_sample_FASTA_quick(fasta_path)
    # af_sample_FASTA_quick(fasta_path1)
    # af_sample_FASTA_quick(fasta_path2)
    # af_sample_FASTA_quick(fasta_path3)
    af_sample_FASTA_quick(fasta_path4)
    # af_sample_FASTA_quick(fasta_path4)
    # af_sample_FASTA_quick(fasta_path2)




    print("Done")
