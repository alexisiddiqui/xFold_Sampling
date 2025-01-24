import json
import os

# import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_directory(output_dir):
    json_info = {}
    file_list = os.listdir(output_dir)
    json_list = [file for file in file_list if file.endswith(".json") and "rank" in file]

    for idx, json_file in enumerate(json_list):
        with open(os.path.join(output_dir, json_file), "r") as file:
            json_data = json.load(file)
        # Select plddt, max_pae, and ptm, and compute average plddt
        json_data = {key: json_data[key] for key in ["plddt", "max_pae", "ptm"] if key in json_data}
        json_data["plddt"] = round(sum(json_data["plddt"]) / len(json_data["plddt"]), 2)

        json_info[idx] = json_data
    return json_info


def combine_af_json_info(af_dir: str):
    dir_names = os.listdir(af_dir)
    dir_names = [
        dir_name for dir_name in dir_names if os.path.isdir(os.path.join(af_dir, dir_name))
    ]
    dir_names.sort(
        key=lambda x: int(x.split("_")[1]), reverse=True
    )  # Sort by max MSA size in descending order

    all_json_info = {}

    # Use ThreadPoolExecutor to parallelize directory processing
    with ThreadPoolExecutor() as executor:  # Adjust number of workers as needed
        future_to_dir = {
            executor.submit(process_directory, os.path.join(af_dir, dir_name)): dir_name
            for dir_name in dir_names
        }

        for future in as_completed(future_to_dir):
            dir_name = future_to_dir[future]
            try:
                all_json_info[dir_name] = future.result()
            except Exception as exc:
                print(f"{dir_name} generated an exception: {exc}")

    # save the json info
    json_name = af_dir + "_ranks.json"
    json_path = os.path.join(af_dir, json_name)

    # order the dictionary by the max MSA size in descending order
    all_json_info = {k: all_json_info[k] for k in dir_names}

    with open(json_path, "w") as file:
        json.dump(all_json_info, file)
    print(f"json_info: {all_json_info}")
    print(f"json_path: {json_path}")
    json_keys = all_json_info.keys()

    print(f"json_keys: {json_keys}")
    print(dir_names)
    return all_json_info


if __name__ == "__main__":
    # test_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_0_af_sample_127_10000"
    # # json_info = combine_af_json_info(test_dir)
    # test_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_1_af_sample_127_10000"
    # json_info = combine_af_json_info(test_dir)
    # test_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_2_af_sample_127_10000"
    # json_info = combine_af_json_info(test_dir)
    # test_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling3/af_sample/BPTI/BPTI/P00974_60_3_af_sample_127_10000"
    # json_info = combine_af_json_info(test_dir)

    # test_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling3/af_sample/BRD4/BRD4/BRD4_APO_484_1_af_sample_127_10000"
    # json_info = combine_af_json_info(test_dir)

    # test_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling3/af_sample/HOIP/HOIP/HOIP_apo697_1_af_sample_127_10000"
    # json_info = combine_af_json_info(test_dir)

    # test_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling3/af_sample/LXRa/LXRa/LXRa200_1_af_sample_127_10000"
    # json_info = combine_af_json_info(test_dir)

    # test_dir = "/homes/hussain/hussain-simulation_hdx/projects/xFold_Sampling3/af_sample/MBP/MBP/MBP_wt_1_af_sample_127_10000"
    # json_info = combine_af_json_info(test_dir)

    test_dir = (
        "/home/alexi/Documents/xFold_Sampling/af_sample/HOIP_dab3/HOIP_dab3_1_af_sample_21_100"
    )
    json_info = combine_af_json_info(test_dir)

    test_dir = (
        "/home/alexi/Documents/xFold_Sampling/af_sample/HOIP_dab3/HOIP_dab3_3_af_sample_21_100"
    )
    json_info = combine_af_json_info(test_dir)
