import os 
import MDAnalysis as mda
import json
import threading
from operator import itemgetter

def dir_to_xtc_ordered(json_path: str,
                       input_dir: str,
                       output_dir: str = None,
                       residue_range: tuple = (None, None)):
    """
    input_dir: str, path to directory containing PDBs
    output_dir: str, path to directory to save xtc files
    uses the name of the tail directory as the name of the xtc file
    """
    if output_dir is None:
        output_dir = os.path.join(*input_dir.split(os.sep)[:-1])
    print(f"Output directory not specified, using {output_dir}")
    
    # make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # sort json data dictionary by splitting the key by "_" and sorting by the 1st element in descending order
    json_data = dict(sorted(json_data.items(), key=lambda x: int(x[0].split("_")[1]), reverse=True))
    print(json_data.keys())

    # get list of PDBs
    pdbs_list = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]
    pdbs = []
    lock = threading.Lock()

    def process_pdb(rank_pdb):
        # Process each pdb in a separate thread
        # Add the pdb to the pdbs list
        with lock:
            pdbs.append(rank_pdb)

    threads = []
    for msa, run in json_data.items():
        msa_pdbs = [pdb for pdb in pdbs_list if msa in pdb]
        msa_pdbs = sorted(msa_pdbs, key=lambda x: int(x.split("_")[-7]))
        for rank_pdb in msa_pdbs:
            thread = threading.Thread(target=process_pdb, args=(rank_pdb,))
            thread.start()
            threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print(f"Found {len(pdbs)} PDBs in {input_dir}")
    pdb_paths = [os.path.join(input_dir, pdb) for pdb in pdbs]
    print(pdb_paths[0])

    # add to universe
    u = mda.Universe(pdb_paths[0], pdb_paths[:])

    # if any residue range is specified, select the atoms
    if residue_range[0] is not None:
        u.atoms = u.atoms[u.atoms.resids >= residue_range[0]]
    if residue_range[1] is not None:
        u.atoms = u.atoms[u.atoms.resids <= residue_range[1]]

    # set resid to start at 1
    u.atoms.residues.resids -= u.atoms.residues.resids.min() - 1

    print(f"Universe has {u.atoms.n_atoms} atoms")
    print(f"Universe has {u.trajectory.n_frames} frames")

    # Collect pLDDT scores for all frames
    plddt_scores = []
    total_frame_count = 0
    for msa, run in json_data.items():
        for frame, frame_data in run.items():
            plddt_score = frame_data["plddt"]
            plddt_scores.append((total_frame_count, plddt_score))
            total_frame_count += 1

    # Sort frames by pLDDT score in descending order
    sorted_frames = sorted(plddt_scores, key=itemgetter(1), reverse=True)

    # Find the frame with the highest pLDDT score
    highest_plddt_frame, highest_plddt_score = sorted_frames[0]

    # Write original xtc
    xtc_name = os.path.basename(input_dir) + '.xtc'
    xtc_path = os.path.join(output_dir, xtc_name)
    print("Writing original xtc to", xtc_path)
    with mda.Writer(xtc_path, u.atoms.n_atoms) as W:
        for idx, ts in enumerate(u.trajectory):
            print(f"Writing frame {idx} of {u.trajectory.n_frames}", end='\r')
            W.write(u.atoms)
    print("\nFinished writing original xtc")

    # Write pLDDT-ordered xtc
    ordered_xtc_name = os.path.basename(input_dir) + '_plddt_ordered.xtc'
    ordered_xtc_path = os.path.join(output_dir, ordered_xtc_name)
    print("Writing pLDDT-ordered xtc to", ordered_xtc_path)
    with mda.Writer(ordered_xtc_path, u.atoms.n_atoms) as W:
        for idx, (frame, _) in enumerate(sorted_frames):
            print(f"Writing frame {idx} of {u.trajectory.n_frames}", end='\r')
            u.trajectory[frame]
            W.write(u.atoms)
    print("\nFinished writing pLDDT-ordered xtc")

    # Write pLDDT information to text file
    plddt_info_name = os.path.basename(input_dir) + '_plddt_info.txt'
    plddt_info_path = os.path.join(output_dir, plddt_info_name)
    print("Writing pLDDT information to", plddt_info_path)
    with open(plddt_info_path, 'w') as f:
        f.write("Frame\tpLDDT\n")
        for frame, plddt in sorted_frames:
            f.write(f"{frame}\t{plddt}\n")
    print("Finished writing pLDDT information")

    # Write the first frame PDB
    pdb_name = os.path.basename(input_dir) + '_first_frame.pdb'
    pdb_path = os.path.join(output_dir, pdb_name)
    print("Writing first frame pdb to", pdb_path)
    u.atoms.write(pdb_path, frames=[0])

    # Write the highest pLDDT frame PDB
    highest_plddt_pdb_name = os.path.basename(input_dir) + f'_max_plddt_{highest_plddt_frame}.pdb'
    highest_plddt_pdb_path = os.path.join(output_dir, highest_plddt_pdb_name)
    print(f"Writing highest pLDDT frame ({highest_plddt_frame}) pdb to", highest_plddt_pdb_path)
    u.atoms.write(highest_plddt_pdb_path, frames=[highest_plddt_frame])

    print("Finished writing pdb files")

if __name__ == "__main__":
    # json_path = "projects/xFold_Sampling/af_sample/BPTI/P00974_60_1_af_sample_10000_ranks.json"

    # input_dir = "projects/xFold_Sampling/af_sample/BPTI/P00974_60_1_af_sample_10000_protonated"
    # # output_dir = "projects/xFold_Sampling/af_sample/BPTI"
    # # dir_to_xtc(input_dir, output_dir, residue_range=(36, 93))


    # input_dir1 = "/home/alexi/Documents/colabquicktest/af_sample/MBP/MBP_wt_protonated"
    # output_dir1 = "/home/alexi/Documents/colabquicktest/af_sample/MBP"

    # dir_to_xtc(input_dir1, output_dir1)



#     input_dir3 = "/home/alexi/Documents/colabquicktest/af_sample/HOIP/HOIP_apo_protonated"
#     output_dir3 = "/home/alexi/Documents/colabquicktest/af_sample/HOIP"

#     # dir_to_xtc(input_dir3, output_dir3)
# # 
#     # input_dir4 = "/data/chem-cat/lina4225/xFold_Sampling/af_sample/BRD4/BRD4_APO_484_af_sample_1000_protonated"
#     # output_dir4 = "/data/chem-cat/lina4225/xFold_Sampling/af_sample/BRD4"

    json_path = "projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_0_af_sample_127_10000_ranks.json"

    input_dir = "projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_0_af_sample_127_10000_protonated"

    dir_to_xtc_ordered(json_path, input_dir)

    json_path = "projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_1_af_sample_127_10000_ranks.json"

    input_dir = "projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_1_af_sample_127_10000_protonated"

    dir_to_xtc_ordered(json_path, input_dir)

    json_path = "projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_2_af_sample_127_10000_ranks.json"

    input_dir = "projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_2_af_sample_127_10000_protonated"

    dir_to_xtc_ordered(json_path, input_dir)

    json_path = "projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_3_af_sample_127_10000_ranks.json"

    input_dir = "projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_3_af_sample_127_10000_protonated"

    dir_to_xtc_ordered(json_path, input_dir)




#     # json_path = "projects/xFold_Sampling/af_sample/BPTI/P00974_60_0_af_sample_10000_ranks.json"

    # input_dir = "projects/xFold_Sampling/af_sample/BPTI/P00974_60_0_af_sample_10000_protonated"

    # dir_to_xtc_ordered(json_path, input_dir)


    # json_path = "projects/xFold_Sampling/af_sample/BPTI/P00974_60_2_af_sample_10000_ranks.json"

    # input_dir = "projects/xFold_Sampling/af_sample/BPTI/P00974_60_2_af_sample_10000_protonated"

    # dir_to_xtc_ordered(json_path, input_dir)


    # json_path = "projects/xFold_Sampling/af_sample/BPTI/P00974_60_3_af_sample_10000_ranks.json"

    # input_dir = "projects/xFold_Sampling/af_sample/BPTI/P00974_60_3_af_sample_10000_protonated"

    # dir_to_xtc_ordered(json_path, input_dir)


    json_path = "projects/xFold_Sampling3/af_sample/BRD4/BRD4/BRD4_APO_484_1_af_sample_127_10000_ranks.json"
    input_dir = "projects/xFold_Sampling3/af_sample/BRD4/BRD4/BRD4_APO_484_1_af_sample_127_10000_protonated"
    dir_to_xtc_ordered(json_path, input_dir)

    json_path = "projects/xFold_Sampling3/af_sample/HOIP/HOIP/HOIP_apo697_1_af_sample_127_10000_ranks.json"
    input_dir = "projects/xFold_Sampling3/af_sample/HOIP/HOIP/HOIP_apo697_1_af_sample_127_10000_protonated"
    dir_to_xtc_ordered(json_path, input_dir)
    json_path = "projects/xFold_Sampling3/af_sample/LXRa/LXRa/LXRa200_1_af_sample_127_10000_ranks.json"
    input_dir = "projects/xFold_Sampling3/af_sample/LXRa/LXRa/LXRa200_1_af_sample_127_10000_protonated"
    dir_to_xtc_ordered(json_path, input_dir)

    json_path = "projects/xFold_Sampling3/af_sample/MBP/MBP/MBP_wt_1_af_sample_127_10000_ranks.json"
    input_dir = "projects/xFold_Sampling3/af_sample/MBP/MBP/MBP_wt_1_af_sample_127_10000_protonated"
    dir_to_xtc_ordered(json_path, input_dir)





    print("Finished all directories")



