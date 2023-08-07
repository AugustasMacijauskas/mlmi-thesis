import argparse
import json
from pathlib import Path


dataset_to_result_key = {
    # "ag_news": "acc,none",
    "ag_news_binarized": "acc,none",
    "amazon_polarity": "acc,none",
    "boolq": "acc,none",
    "boolq_custom": "acc,none",
    # "copa": "acc",
    "copa": "acc,none",
    "dbpedia_14": "acc,none",
    "dbpedia_14_binarized": "acc,none",
    "imdb": "acc,none",
    "imdb_ps3": "acc,none",
    "imdb_ps4": "acc,none",
    "imdb_burns_1": "acc,none",
    "imdb_burns_2": "acc,none",
    # "qnli": "acc,none",
    "qnli_custom": "acc,none",
    "qnli_custom_2": "acc,none",
    "qnli_vicuna": "acc,none",
    "rte_custom": "acc,none",
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
    files = list(output_folder.glob("outputs_burns/*.jsonl"))
    files = [x for x in files if not "gpt" in x.name] # TODO: remove this dirty hack
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
        dataset_name = file.stem
        result = json_dict["results"][dataset_name][dataset_to_result_key[dataset_name]]
        
        key = dataset_name.replace("_custom", "")
        outputs[key] = result * 100
    
    print(json.dumps(outputs, indent=4))

    markdown_outputs = dict_to_markdown(outputs)
    print(markdown_outputs)

    identifier = output_folder.name.split("_")[-1]
    summary_path = output_folder / f"results-summary-burns-{identifier}.md"
    print(summary_path)
    with open(summary_path, "w") as file:
        file.write(markdown_outputs)
    
    return outputs


if __name__ == "__main__":
    main()