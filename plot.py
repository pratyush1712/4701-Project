import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def get_data():
    with open("control.json", "r") as f_control:
        control_data = json.load(f_control)
    with open("condition.json", "r") as f_condition:
        condition_data = json.load(f_condition)
    return control_data, condition_data


def plot_scores_data(condition_data):
    madrs_first = [
        data["madrs1"]
        for _, data in condition_data.items()
        if data["madrs1"] is not None
    ]
    madrs_second = [
        data["madrs2"]
        for _, data in condition_data.items()
        if data["madrs2"] is not None
    ]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.hist(madrs_first, alpha=0.5, label="MADRS 1", bins=range(0, 60, 5))
    plt.hist(madrs_second, alpha=0.5, label="MADRS 2", bins=range(0, 60, 5))
    plt.title("Distribution of MADRS Scores in Condition Group")
    plt.xlabel("MADRS Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("graphs/madrs_scores_distribution.png")
    plt.close()


def flatten_sleep_data(person_data):
    flat_data = [
        {
            "id": person_data["number"],
            "days": person_data["days"],
            "gender": person_data["gender"],
            "age": person_data["age"],
            "afftype": person_data.get("afftype", "Unknown"),
            "melanch": person_data.get("melanch", "Unknown"),
            "inpatient": person_data.get("inpatient", "Unknown"),
            "edu": person_data.get("edu", "Unknown"),
            "marriage": person_data.get("marriage", "Unknown"),
            "work": person_data.get("work", "Unknown"),
            "madrs1": person_data.get("madrs1", "Unknown"),
            "madrs2": person_data.get("madrs2", "Unknown"),
            "date": record["date"],
            "timestamp": record["timestamp"],
            "activity": record["activity"],
            "group": person_data.get("afftype", "Unknown"),
        }
        for record in person_data["sleep-data"]
    ]
    return pd.DataFrame(flat_data)


def aggregate_group_data(group_json):
    all_data = []
    for person_id, person_data in group_json.items():
        person_df = flatten_sleep_data(person_data)
        all_data.append(person_df)

    aggregated_df = pd.concat(all_data, ignore_index=True)
    return aggregated_df


def get_hour_minute(timestamp):
    """Extract hour and minute from timestamp."""
    return timestamp.hour * 60 + timestamp.minute


def plot_individual_activity_patterns(condition_df):
    """Plot daily activity patterns for each user."""
    for person_id in condition_df["id"].unique():
        plt.figure(figsize=(12, 6))
        condition_df.loc[condition_df["id"] == person_id, "minute_of_day"] = (
            condition_df.loc[condition_df["id"] == person_id, "timestamp"].apply(
                get_hour_minute
            )
        )

        avg_activity = (
            condition_df.loc[condition_df["id"] == person_id]
            .groupby("minute_of_day", as_index=False)["activity"]
            .mean()
        )
        sns.lineplot(data=avg_activity, x="minute_of_day", y="activity")

        plt.title(f"Daily Activity Pattern for User {person_id}")
        user_info = (
            f"Age: {condition_df.loc[condition_df['id'] == person_id, 'age'].values[0]}, "
            f"MADRS1: {condition_df.loc[condition_df['id'] == person_id, 'madrs1'].values[0]}, "
            f"MADRS2: {condition_df.loc[condition_df['id'] == person_id, 'madrs2'].values[0]}, "
            f"Melancholia: {condition_df.loc[condition_df['id'] == person_id, 'melanch'].values[0]}"
        )
        plt.legend([user_info])
        plt.ylabel("Activity")
        plt.xlabel("Time (minutes after midnight)")
        plt.savefig(f"graphs/user_{person_id}_daily_activity_pattern.png")
        plt.close()


def plot_group_activity_pattern(condition_df):
    """Plot overall daily activity pattern."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=condition_df, x="timestamp", y="activity", err_style=None)
    plt.title("Daily Activity Patterns")
    plt.ylabel("Activity")
    plt.xlabel("Time")
    plt.savefig("graphs/daily_activity_patterns.png")
    plt.close()


def plot_daily_activity_patterns(condition_data):
    print("Aggregating group data...")
    condition_df = aggregate_group_data(condition_data)
    condition_df["timestamp"] = pd.to_datetime(condition_df["timestamp"])

    try:
        print("Plotting daily activity patterns for each user...")
        plot_individual_activity_patterns(condition_df)

        print("Plotting overall daily activity pattern...")
        plot_group_activity_pattern(condition_df)

        print("Plotting completed. Graphs saved.")
    except Exception as e:
        print(f"Error: {e}")
        print("Unable to plot daily activity patterns.")


if __name__ == "__main__":
    control_data, condition_data = get_data()
    print("Data loaded successfully.")
    plot_scores_data(condition_data)
    print("MADRS scores plotted.")
    plot_daily_activity_patterns(condition_data)
    plot_daily_activity_patterns(control_data)
