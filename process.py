import pandas as pd
import json
import os


CONTROL_DATA = "data/control"
CONDITION_DATA = "data/condition"
SCORES = "data/scores.csv"


def process_patient_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def aggregate_group_data(group_path, group_name, scores_df, aggregated_data):
    for filename in os.listdir(group_path):
        if filename.endswith(".csv"):
            patient_id = filename.replace(".csv", "").strip()
            if patient_id in scores_df["number"].values:
                patient_data = scores_df[scores_df["number"] == patient_id].to_dict(
                    orient="records"
                )[0]
                for key, value in patient_data.items():
                    if pd.isna(value):
                        if isinstance(value, float):
                            patient_data[key] = None
                        elif value == " ":
                            patient_data[key] = "Unknown"
            else:
                patient_data = {
                    "number": patient_id,
                    "afftype": None,
                    "melanch": None,
                    "inpatient": None,
                    "edu": "Unknown",
                    "marriage": None,
                    "work": None,
                    "madrs1": None,
                    "madrs2": None,
                }
            patient_data["sleep-data"] = process_patient_csv(
                os.path.join(group_path, filename)
            )
            aggregated_data[group_name][patient_id] = patient_data


def aggregate_and_convert_to_json(control_path, condition_path, scores_path):
    scores_df = pd.read_csv(scores_path)
    aggregated_data = {"control": {}, "condition": {}}

    aggregate_group_data(control_path, "control", scores_df, aggregated_data)
    aggregate_group_data(condition_path, "condition", scores_df, aggregated_data)

    with open("control.json", "w") as f_control:
        json.dump(aggregated_data["control"], f_control, indent=4)
    with open("condition.json", "w") as f_condition:
        json.dump(aggregated_data["condition"], f_condition, indent=4)


if __name__ == "__main__":
    aggregate_and_convert_to_json(CONTROL_DATA, CONDITION_DATA, SCORES)
