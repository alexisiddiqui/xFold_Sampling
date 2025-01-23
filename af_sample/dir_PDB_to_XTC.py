# turn dirs of PDBs into xtc 

import os 
import MDAnalysis as mda



def dir_to_xtc(input_dir:str,
               output_dir:str=None,
               residue_range:tuple = (None, None)):
    """
    input_dir: str, path to directory containing PDBs
    output_dir: str, path to directory to save xtc files
    uses the name of the tail directory as the name of the xtc file
    """
    if output_dir is None:
        output_dir = input_dir.split(os.sep)[:-1]
        output_dir = os.path.join(*output_dir)
        print(f"Output directory not specified, using {output_dir}")    

    # make output directory if it doesn't exist
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # get list of PDBs
    pdbs = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]
    print(f"Found {len(pdbs)} PDBs in {input_dir}")
    pdb_paths = [os.path.join(input_dir, pdb) for pdb in pdbs]
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

    # write xtc
    xtc_name = os.path.basename(input_dir) + '.xtc'
    xtc_path = os.path.join(output_dir, xtc_name)
    print("Writing xtc to", xtc_path)
    with mda.Writer(xtc_path, u.atoms.n_atoms) as W:
        for idx, ts in enumerate(u.trajectory):
            print(f"Writing frame {idx} of {u.trajectory.n_frames}", end='\r')
            W.write(u.atoms)

    
    print("Finished writing xtc")

    pdb_name = os.path.basename(input_dir) + '.pdb'
    pdb_path = os.path.join(output_dir, pdb_name)

    print("Writing pdb to", pdb_path)
    u.atoms.write(pdb_path)
    print("Finished writing pdb topology")




if __name__ == "__main__":
    json_path = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling/af_sample/BPTI/P00974_60_1_af_sample_10000_ranks.json"

    input_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling/af_sample/BPTI/P00974_60_1_af_sample_10000_protonated"
    # output_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling/af_sample/BPTI"
    # dir_to_xtc(input_dir, output_dir, residue_range=(36, 93))


    input_dir1 = "/home/alexi/Documents/colabquicktest/af_sample/MBP/MBP_wt_protonated"
    output_dir1 = "/home/alexi/Documents/colabquicktest/af_sample/MBP"

    # dir_to_xtc(input_dir1, output_dir1)

    input_dir2 = "/home/alexi/Documents/colabquicktest/af_sample/LXRa/LXRa_protonated"
    output_dir2 = "/home/alexi/Documents/colabquicktest/af_sample/LXRa"

    # dir_to_xtc(input_dir2, output_dir2)

    input_dir3 = "/home/alexi/Documents/colabquicktest/af_sample/HOIP/HOIP_apo_protonated"
    output_dir3 = "/home/alexi/Documents/colabquicktest/af_sample/HOIP"

    # dir_to_xtc(input_dir3, output_dir3)
# 
    input_dir4 = "/data/chem-cat/lina4225/xFold_Sampling/af_sample/BRD4/BRD4_APO_484_af_sample_1000_protonated"
    output_dir4 = "/data/chem-cat/lina4225/xFold_Sampling/af_sample/BRD4"



    dir_to_xtc(input_dir)

    print("Finished all directories")

