import os
import subprocess
import json
import sys  # Import the sys module

def af_sample_FASTA_quick(fasta_path:str,
                          output_dir:str=None,
                          num_seeds:int=200,
                          max_seq_range: tuple[int,int] = (2,256),
                          max_models = 10000, 
                          use_dropout:bool = True,
                          num_recycle:int =1):

    num_af_models = 5
    if output_dir is None:
        output_dir = fasta_path.split(".")[0] + f"_{num_recycle}_af_sample_{str(max_models)}"

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

    json_info = {}

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
        try:
            subprocess.run(alphafold_command, check=True)
        except:
            subprocess.run(' '.join(alphafold_command), check=True, shell=True)

        finally:
            # json list 
            file_list = os.listdir(output_dir)
            json_list = [file for file in file_list if file.endswith(".json") and "rank" in file]


            for idx, json_file in enumerate(json_list):
                json_info[max_MSA_string] = {}
                with open(os.path.join(output_dir, json_file), "r") as file:
                    json_data = json.load(file)
                    # select plddt and max_pae adn ptm
                    json_data = {key: json_data[key] for key in ["plddt", "max_pae", "ptm"]}
                    json_data["plddt"] = sum(json_data["plddt"])/len(json_data["plddt"])
                    json_info[max_MSA_string][idx] = json_data


    # save the json info
    json_name = output_dir+"_ranks.json"
    with open(json_name, "w") as file:
        json.dump(json_info, file)





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <fasta_path>")
    else:
        fasta_path = sys.argv[1]  # Get fasta path from command-line argument
        # Optionally, you can add more command-line arguments to customize other parameters like num_recycle
        af_sample_FASTA_quick(fasta_path)
        print("Done")

