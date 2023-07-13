import argparse
import json
from pathlib import Path


dataset_to_result_key = {
    "arc_challenge_json": "acc_norm",
    "arc_challenge_jsonl": "acc_norm,none",
    "hellaswag_json": "acc_norm",
    "hellaswag_jsonl": "acc_norm,none",
    "mmlu_json": "acc_norm,none",
    "truthfulqa_mc_json": "mc2",
    "qnli_jsonl": "acc,none",
}


def main():
    # Read the folder name from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    
    # Get the files in the output folder
    output_folder = Path("/fsx/home-augustas/" + args.output_path)
    files = list(output_folder.glob("outputs/*-shot.json"))
    files.extend(list(output_folder.glob("outputs/*-shot.jsonl")))
    files.extend(list(output_folder.glob("outputs/qnli.jsonl")))
    # files.extend(output_folder.glob("outputs/truthfulqa_mc-0-shot.json"))
    # files.extend(output_folder.glob("outputs/mmlu-5-shot.json"))
    files = sorted(files, key=lambda x: x.name)
    print([x.name for x in files])

    # Iterate through the json files and get the results
    outputs = {}
    for file in files:
        # Open the json file
        with open(file, "r") as f:
            json_dict = json.load(f)
        
        print(json_dict["config"]["model_args"])
        
        # Get the results
        dataset_name = file.stem.split("-")[0]
        results_key = f"{dataset_name}_{file.suffix[1:]}"
        if dataset_name == "mmlu":
            result = [json_dict["results"][key]["acc"] for key in json_dict["results"]]
            result = sum(result) / len(result)
        else:
            result = json_dict["results"][dataset_name][dataset_to_result_key[results_key]]
        
        print(result)
        outputs[results_key] = f"{result * 100:.1f}"
    
    print(json.dumps(outputs, indent=4))


if __name__ == "__main__":
    main()