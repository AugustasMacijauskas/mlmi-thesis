import argparse
import pandas as pd
from pathlib import Path
import re


def get_accuracy_from_file(file_path):
    with open(file_path, 'r') as file:
        # Accuracy is on the 2nd line
        accuracy_line = file.readlines()[1].strip()
    
    return float(accuracy_line[:-1])


def dict_to_markdown(outputs):
    ret = ""
    # ret += "| --- | --- |\n"
    ret += "| dataset | accuracy |\n"
    ret += "| --- | --- |\n"

    for dataset, accuracy in outputs.items():
        ret += f"| {dataset} | {accuracy:.2f}% |\n"
    
    # Average
    average = sum(outputs.values()) / len(outputs.values())
    ret += f"| **Average** | **{average:.2f}%** |\n"
    # ret += "| --- | --- |\n"
            
    return ret


def process_results(output_path):
    result_files = list(output_path.glob("results-out-*.md"))

    outputs = {}
    for file_path in result_files:
        dataset_name = file_path.stem.split("-")[2]
        
        accuracy = get_accuracy_from_file(file_path)
        outputs[dataset_name] = accuracy
    
    outputs = dict_to_markdown(outputs)

    identifier = output_path.name.split("_")[-1]

    summary_path = output_path / f"results-summary-{identifier}.md"
    with open(summary_path, "w") as file:
        file.write(outputs)
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Process experimental results.')
    parser.add_argument('--output_path', type=str, help='Path to the folder containing the evalution results')
    parser.add_argument('--suffix', type=str, help='(optional) suffix', default="")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    
    return process_results(output_path)


if __name__ == "__main__":
    main()
