# Activate environment 
conda activate emotional

cd script


JSON_FILE="generated_scenario.json"
DATASET="humaneval"
MODEL="google/codegemma-7b-it"
RESULT_DIR="output_path"

echo "Saving results to $RESULT_DIR"
mkdir -p "$RESULT_DIR"

python3 <<EOF
import json
import subprocess
import os

json_file = "$JSON_FILE"
dataset = "$DATASET"
model = "$MODEL"
out_dir = "$RESULT_DIR"

model_safe = model.replace("/", "_")

with open(json_file, "r") as f:
    data = json.load(f)


# 1. Baseline (NO sentiment)


baseline_output = os.path.join(out_dir, f"results_{model_safe}_{dataset}_baseline.json")
baseline_eval = os.path.join(out_dir, f"eval_{model_safe}_{dataset}_baseline.json")

baseline_cmd = [
    "python", "main.py",
    "--model", model,
    "--dataset", dataset,
    "--output_file", baseline_output,
    "--eval_output_file", baseline_eval,
]

print("Running baseline (no sentiment)...")
print(" ".join(baseline_cmd))
subprocess.run(baseline_cmd)

# 2. Run all sentiment triggers

for item in data:
    trigger = item["trigger"]
    trigger_safe = trigger.replace(" ", "_")

    output_file = os.path.join(out_dir, f"results_{model_safe}_{dataset}_{trigger_safe}.json")
    eval_file = os.path.join(out_dir, f"eval_{model_safe}_{dataset}_{trigger_safe}.json")

    cmd = [
        "python", "main.py",
        "--model", model,
        "--dataset", dataset,
        "--sentiment_trigger", trigger,   
        "--emotion_json", json_file,      
        "--output_file", output_file,
        "--eval_output_file", eval_file,
    ]

    print(f"Running Trigger: {trigger}")
    print(" ".join(cmd))
    subprocess.run(cmd)
EOF
