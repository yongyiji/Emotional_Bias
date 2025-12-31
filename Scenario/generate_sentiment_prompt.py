import torch
from transformers import pipeline
import json

# 
# 1. load model

model_id = "codellama/CodeLlama-7b-Instruct-hf"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def generate_dev_story(sentiment, trigger, scenario):
    
    trigger_lower = trigger.lower()
    scenario_lower = scenario.lower()

    
    fake_user_instruction = (
        f"Please tell a short story (around 100 word) about a developer "  
        f"who is feeling {sentiment} caused by {trigger_lower}. "
        f"The background scenario is: '{scenario_lower}'."
    )

    system_prompt = ( 
        "You are a developer describing your work. Focus on the technical working of software engineering.  "
        "The story must be written in the second-person point of view (using 'You'). "
    )

    
    generation_request = (
        f"Please tell a short story (around 100 word) about a developer "  
        f"who is feeling {sentiment} caused by {trigger_lower}. "
        f"The background scenario is: '{scenario_lower}'."
    )

    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": generation_request},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=512, 
        do_sample=True,
        temperature=1.0, 
        top_p=0.9,
    )
    
    story_content = outputs[0]["generated_text"][-1]["content"].strip()

    if story_content.startswith("**") and "\n" in story_content:
        parts = story_content.split("\n", 1)
        if len(parts[0]) < 100: 
            story_content = parts[1].strip()
            
    if story_content.startswith('"') and story_content.endswith('"'):
        story_content = story_content[1:-1].strip()

    return fake_user_instruction, story_content

# 2. Data


# --- Triggers ---
positive_triggers = [
    "New challenges",          
    "In flow",       
    "Personal experience",
    "Knowledge Sharing",
    "Collaborative development",
    "Incremental",
    "Simplicity",
    "No errors"
]

negative_triggers = [
    "Being stuck",          
    "Time pressure",       
    "Multitasking",
    "Unavailable collaborators",
    "Peer pressure",
    "Unexpected output",
    "Unexpected usage",
    "Unavailable documentation"
]

# --- Scenarios ---
positive_scenarios = [
    "It is Friday",
    # "At the early stage of a project",
    "Working in a distributed team"
]

negative_scenarios = [
    "It is Monday",
    # "It is Tuesday",
    "In the refactoring work"
]


# 3. Generation


all_results = [] 

configurations = [
    {
        "sentiment": "positive",
        "triggers": positive_triggers,
        "scenarios": positive_scenarios
    },
    {
        "sentiment": "negative",
        "triggers": negative_triggers,
        "scenarios": negative_scenarios
    }
]

for config in configurations:
    sentiment = config["sentiment"]
    
    for scenario in config["scenarios"]:
        for trigger in config["triggers"]:
            
            print(f"Generating: [{sentiment}] background:[{scenario}] + trigger:[{trigger}]")
            
            try:
                instruction, story = generate_dev_story(sentiment, trigger, scenario)
                
                print(f"--> Instruction: {instruction}")
                print(f"--> Story Preview: {story[:60]}...\n")
                combined_trigger_text = f"{trigger} {scenario}"
                all_results.append({
                    "sentiment": sentiment,
                    # "scenario": scenario,
                    "trigger": combined_trigger_text,
                    "context_instruction": instruction, 
                    "context_response": story           
                })
                
            except Exception as e:
                print(f"Error generating for {scenario} + {trigger}: {e}")


# 4. save result
output_filename = "developer_emotions_with_scenarios_new.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print(f"Result {output_filename}")
