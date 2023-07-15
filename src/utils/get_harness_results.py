import argparse
import json
from pathlib import Path


dataset_to_result_key = {
    "arc_challenge": "acc_norm,none",
    # "arc_challenge_json": "acc_norm",
    # "arc_challenge_jsonl": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    # "hellaswag_json": "acc_norm",
    # "hellaswag_jsonl": "acc_norm,none",
    "mmlu": "acc_norm,none",
    "truthfulqa_mc": "mc2",
}


def dict_to_markdown(outputs):
    ret = ""
    # ret += "| --- | --- |\n"
    ret += "| dataset | accuracy |\n"
    ret += "| --- | --- |\n"

    for dataset in sorted(outputs.keys()):
        accuracy = outputs[dataset]
        ret += f"| {dataset} | {accuracy:.2f} |\n"
    
    # Average
    average = sum(outputs.values()) / len(outputs.values())
    ret += f"| **Average** | **{average:.2f}** |\n"
    # ret += "| --- | --- |\n"
            
    return ret


def main():
    # Read the folder name from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    
    # Get the files in the output folder
    output_folder = Path("/fsx/home-augustas/" + args.output_path)
    files = list(output_folder.glob("outputs_open_llm/*-shot.jsonl"))
    files.extend(output_folder.glob("outputs_open_llm/*-shot.json"))
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
        if dataset_name == "mmlu":
            result = [json_dict["results"][key]["acc"] for key in json_dict["results"]]
            result = sum(result) / len(result)
        else:
            result = json_dict["results"][dataset_name][dataset_to_result_key[dataset_name]]
        
        outputs[dataset_name] = result * 100
    
    print(json.dumps(outputs, indent=4))

    markdown_outputs = dict_to_markdown(outputs)
    print(markdown_outputs)

    identifier = output_folder.name.split("_")[-1]
    summary_path = output_folder / f"results-summary-open-llm-{identifier}.md"
    print(summary_path)
    with open(summary_path, "w") as file:
        file.write(markdown_outputs)
    
    return outputs


if __name__ == "__main__":
    main()