import argparse
import pandas as pd
from pathlib import Path
import re


COLUMNS_TO_EXTRACT_DICT = {
    "acc_estimate": "Accuracy",
    "auroc_estimate": "AUROC",
    "cal_acc_estimate": "Calibrated accuracy",
}
RESULTS_TO_EXTRACT = ["eval", "lr_eval", "lm_eval"]


def get_last_line(file_path):
    with open(file_path, 'r') as file:
        last_line = file.readlines()[-1].strip()
    
    last_line = last_line.split(" ")[-1][4:-4]

    return Path(last_line)


def get_max_values_for_column(csv_file, column):
    df = pd.read_csv(csv_file)

    max_value = df[column].max()
    max_value_rows = df[df[column] == max_value]

    results = []
    for _, row in max_value_rows.iterrows():
        entry = row[["layer", "ensembling"]].to_dict()
        entry["value"] = row[column]
        results.append(entry)

    return results


def dict_to_markdown(outputs):
    ret = ""
    for metric, values in outputs.items():
        ret += f"# {metric}\n\n"
        ret += "| lm_eval | eval | lr_eval |\n"
        ret += "| --- | --- | --- |\n"
        ret += "| "

        for idx, value in enumerate(values.values()):
            formatted_value = f"{value[-1]['value'] * 100:.2f}%"
            if metric == "AUROC":
                formatted_value = f"{value[-1]['value']:.4f}"
            
            ret += formatted_value
            if idx < len(values.values()) - 1:
                ret += " | "
            else:
                ret += " |\n"

    ret += "\n"
    for metric, values in outputs.items():
        ret += f"# {metric}\n\n"

        for method, value in values.items():
            ret += f"## {method}\n\n"
            ret += "| Layer | Ensembling | Value |\n"
            ret += "| --- | --- | --- |\n"
            for entry in value:
                formatted_value = f"{entry['value'] * 100:.2f}%"
                if metric == "AUROC":
                    formatted_value = f"{entry['value']:.4f}"
                ret += f"| {entry['layer']} | {entry['ensembling']} | {formatted_value} |\n"
            
    return ret

def process_results(file_path):
    csv_file = get_last_line(file_path)

    outputs = {}
    for column_name, column_label in COLUMNS_TO_EXTRACT_DICT.items():
        new_entry = {}
        for file_stem in RESULTS_TO_EXTRACT:
            results = get_max_values_for_column(csv_file / f"{file_stem}.csv", column=column_name)
            new_entry[file_stem] = results
        
        outputs[column_label] = new_entry

    outputs = dict_to_markdown(outputs)
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Process experimental results.')
    parser.add_argument('--file_path', type=str, help='Path to the file containing the last line with CSV path')
    args = parser.parse_args()

    file_path = Path(args.file_path)
    
    return process_results(file_path)


if __name__ == "__main__":
    main()
