# convert directories from AF sample to protonated topology and trajectory
# select residu range


import os
import subprocess
import numpy as np
import MDAnalysis as mda
from concurrent.futures import ProcessPoolExecutor
import datetime

def protonate_backbone_PDB(input_pdb:str,
                           output_pdb:str,
                           residue_range=tuple):

    H_bond_length = 1.02 # in Angstroms

    # first grab all the backbone nitrogen coordinates
    u = mda.Universe(input_pdb)

    first_resid = u.residues.resids[0]

    print(len(u.residues))
    residues = u.select_atoms(f'not resid {first_resid}').residues
    print(residues.resids)
    last_resid = residues.resids[-1]

    N = u.select_atoms(f'name N and not resid {first_resid}')
    CA = u.select_atoms(f'name CA and not resid {first_resid}')
    C = u.select_atoms(f'name C and not resid {last_resid}')

    N_xyz = N.positions
    print(N_xyz.shape)
    # find the center of the CA and C bonds
    midpoints = (CA.positions + C.positions) / 2

    # find the vector from the midpoint to the N atom
    N_to_midpoint = N_xyz - midpoints

    # normalize the vector
    N_to_midpoint /= np.linalg.norm(N_to_midpoint, axis=1)[:, np.newaxis]

    # multiply the normalized vector by the bond length
    N_to_midpoint *= H_bond_length

    # add the vector to the N atom
    H_xyz = N_xyz + N_to_midpoint

    print("Original universe", u.atoms)

    seg_index = [0]*len(H_xyz)
    
    # create universe from N atoms - this is the universe we will add the hydrogens to
    H_universe = mda.Merge(N)
    print("H universe", H_universe.atoms)

    print(len(H_universe.atoms))

    # replace N information with H information
    H_universe.atoms.positions = H_xyz
    H_universe.atoms.names = "H"
    H_universe.atoms.types = "H"
    H_universe.atoms.elements = "H"

    print("H universe", H_universe.atoms)
    
    # remove PRO residues
    H_universe = H_universe.select_atoms('not resname PRO')


    # repeat for the N terminus
    # will need to condsider how to place H H2 H3


    # merge the H universe with the original universe
    u = mda.Merge(u.atoms, H_universe.atoms)

    # reorder by resid
    u.atoms = u.atoms[u.atoms.resids.argsort()]
    # start the resid at 1

    print("Merged universe", u.atoms)
    #

    # write the new pdb file
    u.atoms.write(output_pdb)

       






def protonate_backbone_AF_dir(input_dir:str,
                              output_dir:str,
                              recursive:bool=False):
    """
    Protonates a directory of AF predicted PDBs adds protons to the backbone Nitrogen.
    This uses a barycentric embedding to compute the vector for the N-H bond.
    H bond length taken from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2670607/#:~:text=Anharmonicity%20of%20the%20bond%20stretching,angular%20fluctuations%20in%20N%2DH%20orientation.
    """

    # check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Protonating directory {input_dir} to {output_dir}")

    if not recursive:
        # get the list of pdb files
        pdb_files = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]

        # make names unique
        

        # loop over the pdb files using concurrent futures
        with ProcessPoolExecutor() as executor:
            for idx, pdb_file in enumerate(pdb_files):
                time_str = datetime.datetime.now().strftime("%d%H%M%S%f")
                input_pdb = os.path.join(input_dir, pdb_file)
                pdb_file = f"{idx}_{time_str}_{pdb_file}"
                output_pdb = os.path.join(output_dir, pdb_file)
                executor.submit(protonate_backbone_PDB, input_pdb, output_pdb)

    if recursive:
        # get the list of directories
        dirs = [d for d in os.listdir(input_dir)]
        print(dirs)
        # loop over the directories using concurrent futures
        with ProcessPoolExecutor() as executor:
            for dir in dirs:
                input_subdir = os.path.join(input_dir, dir)
                # output_subdir = os.path.join(output_dir, dir)
                executor.submit(protonate_backbone_AF_dir, input_subdir, output_dir)


if __name__ == "__main__":
    # test the function
    # input_pdb = "/home/alexi/Documents/colabquicktest/af_sample/HOIP/HOIP_apo_af_sample/maxMSA_13_27/HOIPapo_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_001.pdb"
    # output_pdb = "/home/alexi/Documents/colabquicktest/af_sample/HOIP/HOIP_apo_af_sample/HOIPapo_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_001_protonated.pdb"
    # protonatee_backbone_PDB(input_pdb, output_pdb)

    # input_dir = "/home/alexi/Documents/colabquicktest/af_sample/HOIP/HOIP_apo_af_sample/maxMSA_13_27"
    # output_dir = "/home/alxi/Documents/colabquicktest/af_sample/HOIP/HOIP_apo_af_sample/test_protonated"

    # protonate_backbone_AF_dir(input_dir, output_dir)

    ###

    input_dir = "/home/alexi/Documents/colabquicktest/af_sample/BPTI/P00974_af_sample"
    output_dir = "/home/alexi/Documents/colabquicktest/af_sample/BPTI/P00974_protonated"

    # protonate_backbone_AF_dir(input_dir, output_dir, recursive=True)

    input_dir1 = "/home/alexi/Documents/colabquicktest/af_sample/MBP/MBP_wt_af_sample"
    output_dir1 = "/home/alexi/Documents/colabquicktest/af_sample/MBP/MBP_wt_protonated"

    # protonate_backbone_AF_dir(input_dir1, output_dir1, recursive=True)

    input_dir2 = "/home/alexi/Documents/colabquicktest/af_sample/LXRa/LXRa_af_sample"
    output_dir2 = "/home/alexi/Documents/colabquicktest/af_sample/LXRa/LXRa_protonated"

    # protonate_backbone_AF_dir(input_dir2, output_dir2, recursive=True)

    input_dir3 = "/home/alexi/Documents/colabquicktest/af_sample/HOIP/HOIP_apo_af_sample"
    output_dir3 = "/home/alexi/Documents/colabquicktest/af_sample/HOIP/HOIP_apo_protonated"

    protonate_backbone_AF_dir(input_dir3, output_dir3, recursive=True)

    input_dir4 = "/home/alexi/Documents/colabquicktest/af_sample/BRD4/BRD4_APO_484_af_sample"
    output_dir4 = "/home/alexi/Documents/colabquicktest/af_sample/BRD4/BRD4_APO_484_protonated"

    # protonate_backbone_AF_dir(input_dir4, output_dir4, recursive=True)

