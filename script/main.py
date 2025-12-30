import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from models import load_model
from evaluation import load_standardized_dataset, compute_code_eval
import torch
import gc
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def save_json(data, filename):
    # check the directory
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved results to {filename}")

def main():
    start_time = datetime.now()
    print(f"Process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["humaneval", "humaneval+", "mbpp", "mbpp+", "apps"])
    parser.add_argument("--n_samples", type=int, default=10, help="Must be >= 10 for pass@10")
    parser.add_argument("--output_file", type=str, default="output.json", 
                        help="Path to save the generated code samples (Format: [[code...], [code...]])")
    parser.add_argument("--eval_output_file", type=str, default=None, 
                        help="Optional: Custom path to save the evaluation metrics")
    parser.add_argument("--sentiment_trigger", type=str, default=None,
                        help="Trigger phrase to inject sentiment from JSON")
    parser.add_argument("--emotion_json", type=str, default=None, 
                        help="Path to the developer emotions context JSON file")
    parser.add_argument("--use_cot", type=str, default="false", choices=["true", "false"],
                    help="Whether to use Chain-of-Thought (true/false). Default is false.")
    parser.add_argument("--use_fewshot", type=str, default="false", choices=["true", "false"],
                    help="Whether to use Few-Shot prompting (true/false). Default is false.")
    args = parser.parse_args()

    # 1. load dataset
    print(f"Loading dataset: {args.dataset}...")

    problems, task_ids = load_standardized_dataset(args.dataset)
    print(f"Loaded {len(task_ids)} tasks.")
    
    # 2. load model
    print(f"Loading model: {args.model}...")
    if args.sentiment_trigger:
        print(f"Applied Sentiment Trigger: {args.sentiment_trigger}")
        
    llm = load_model(args.model, sentiment_trigger=args.sentiment_trigger, json_path=args.emotion_json)

    # 3. task
    is_mbpp_task = "mbpp" in args.dataset
    is_apps_task = "apps" in args.dataset
    
    use_chat_format = is_mbpp_task or is_apps_task

    # 4. generate output
    print(f"Generating {args.n_samples} samples per prompt...")
    
    all_predictions = []  
    all_references = []   

    use_cot_bool = args.use_cot.lower() == "true"
    use_fewshot_bool = args.use_fewshot.lower() == "true"

    for task_id in tqdm(task_ids):
        problem = problems[task_id]
        prompt = problem["prompt"]
        

        batch_codes = llm.generate([prompt], n_samples=args.n_samples, is_mbpp=use_chat_format, 
                                    use_cot=use_cot_bool, use_fewshot=use_fewshot_bool, 
                                    dataset_name=args.dataset,
                                    max_new_tokens=1024 if use_cot_bool else 512)[0]
        
        final_candidates = []
        for code in batch_codes:
            if not use_chat_format: 
                cleaned_code = code.strip()
                cleaned_prompt = prompt.strip()
                
                prompt_lines = cleaned_prompt.split('\n')
                import_lines = [line for line in prompt_lines if line.startswith("import ") or line.startswith("from ")]
                import_header = "\n".join(import_lines) + "\n" if import_lines else ""

                if cleaned_prompt in cleaned_code:
                    final_candidates.append(cleaned_code)
                
                elif "def " in cleaned_code:
                    if import_header.strip() and import_header.strip() not in cleaned_code:
                        final_candidates.append(import_header + cleaned_code)
                    else:
                        final_candidates.append(cleaned_code)
                
                else:
                    final_candidates.append(prompt + code)
            else:
                if args.dataset == "apps":
                    cleaned_code = code
                    if "```python" in code:
                        cleaned_code = code.split("```python")[1].split("```")[0]
                    elif "```" in code:
                        cleaned_code = code.split("```")[1].split("```")[0]
                    final_candidates.append(cleaned_code.strip())
                else:
                    final_candidates.append(code)

        all_predictions.append(final_candidates)
        all_references.append(problem["test_code"])
        
        gc.collect()
        
        torch.cuda.empty_cache()
    

    # 5. Evaluation
    print("Running evaluation (Executing code)...")
    
    k_list = [1, 5, 10]
    k_list = [k for k in k_list if k <= args.n_samples]

    if not k_list:
        k_list = [1]

    pass_at_k, detailed_results = compute_code_eval(
        predictions=all_predictions,
        references=all_references,
        k=k_list,
        num_workers=4, 
        timeout=3.0    
    )

    print("\n" + "="*35)
    print(f"Evaluation Results: {args.dataset}")
    if args.sentiment_trigger:
        print(f"Trigger: {args.sentiment_trigger}")
    print(pass_at_k)
    print("="*35 + "\n")

    # 6. save result
    
    # [[sample1_1, sample1_2...], [sample2_1, sample2_2...]]
    save_json(all_predictions, args.output_file)
    
    if args.eval_output_file:
        final_eval_path = args.eval_output_file
    else:
        final_eval_path = args.output_file.replace(".json", "_eval.json")

    end_time = datetime.now()
    duration = end_time - start_time
    
    print("-" * 50)
    print(f"Process ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {duration}")
    print("-" * 50 + "\n")

    save_json({
        "dataset": args.dataset,
        "model": args.model,
        "trigger": args.sentiment_trigger,
        "metrics": pass_at_k,
        "details_sample_count": len(detailed_results),
        "execution_time": str(duration)
    }, final_eval_path)

if __name__ == "__main__":
    main()
