import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(14, 8))

# Define the stages of the workflow
stages = [
    "Data Preprocessing",
    "Model Training",
    "Model Evaluation",
]

# Define the processes within each stage
processes = {
    "Data Preprocessing": ["Process CSV files", "Aggregate Data", "Convert to JSON"],
    "Model Training": ["Random Forest", "XGBoost", "RNN", "SVM"],
    "Model Evaluation": [
        "Evaluate Models using Accuracy, F1, Recall",
        "Confusion Matrix Analysis",
    ],
}

# Define the colors for each stage
colors = {
    "Data Collection": "#ff9999",
    "Data Preprocessing": "#66b3ff",
    "Model Training": "#ffcc99",
    "Model Evaluation": "#c2c2f0",
}

# Position settings
x_pos = 0.1
y_pos = 0.9
width = 0.8
height = 0.05
y_step = 0.06

# Draw the stages and processes
for stage in stages:
    ax.add_patch(
        mpatches.Rectangle(
            (x_pos, y_pos),
            width,
            height,
            linewidth=1,
            edgecolor="black",
            facecolor=colors[stage],
        )
    )
    plt.text(
        x_pos + 0.02, y_pos + height / 2, stage, va="center", ha="left", fontsize=12
    )
    y_pos -= y_step

    for process in processes[stage]:
        ax.add_patch(
            mpatches.Rectangle(
                (x_pos + 0.1, y_pos),
                width - 0.1,
                height,
                linewidth=1,
                edgecolor="black",
                facecolor="white",
            )
        )
        plt.text(
            x_pos + 0.12,
            y_pos + height / 2,
            process,
            va="center",
            ha="left",
            fontsize=10,
        )
        y_pos -= y_step

# Draw arrows between stages
for i in range(len(stages)):
    line = mlines.Line2D(
        [x_pos, x_pos],
        [y_pos + 4 * y_step * i, y_pos + 4 * y_step * (i + 1)],
        color="black",
        linewidth=1.5,
        marker="o",
        markersize=5,
    )
    ax.add_line(line)

# Adjust limits and remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

plt.title(
    "Project Workflow: Predictive Analysis in Mental Health Using Sleep Activity Data",
    fontsize=16,
)
plt.show()
