import argparse
import json
from pathlib import Path


dataset_to_result_key = {
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm,none",
    # "hellaswag": "acc_norm",
    "mmlu": "acc_norm,none",
    "truthfulqa_mc": "mc2",
}


def main():
    # Read the folder name from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    
    # Get the files in the output folder
    output_folder = Path("/fsx/home-augustas/" + args.output_path)
    files = list(output_folder.glob("outputs/*-shot.json"))
    files.extend(output_folder.glob("outputs/mmlu-5-shot.json"))
    files.extend(output_folder.glob("outputs/hellaswag-10-shot.jsonl"))
    files = sorted(files, key=lambda x: x.name)

    # Iterate through the json files and get the results
    outputs = {}
    for file in files:
        # Open the json file
        with open(file, "r") as f:
            json_dict = json.load(f)
        
        # Get the results
        dataset_name = file.name.split("-")[0]
        if dataset_name == "mmlu":
            print(len(json_dict["results"]))
            result = [json_dict["results"][key]["acc_norm"] for key in json_dict["results"]]
            result = sum(result) / len(result)
        else:
            result = json_dict["results"][dataset_name][dataset_to_result_key[dataset_name]]
        
        outputs[dataset_name] = f"{result * 100:.1f}"
    
    print(json.dumps(outputs, indent=4))


if __name__ == "__main__":
    main()