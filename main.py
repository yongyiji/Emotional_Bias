import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from models import load_model
# 引入新的 evaluate 模块
from evaluation import load_standardized_dataset, compute_code_eval
import torch
import gc
from datetime import datetime
# 1. 修复 Tokenizers 警告 (必须在导入 transformers/tokenizers 之前设置)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. 允许代码执行 (HuggingFace 评估库的安全锁)
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def save_json(data, filename):
    # 简单的目录检查，防止路径不存在报错
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved results to {filename}")

def main():
    start_time = datetime.now()
    print(f"\n🚀 [START] Process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    # 添加 'apps' 到支持列表
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["humaneval", "humaneval+", "mbpp", "mbpp+", "apps"])
    parser.add_argument("--n_samples", type=int, default=10, help="Must be >= 10 for pass@10")
    
    # 输出参数
    parser.add_argument("--output_file", type=str, default="output.json", 
                        help="Path to save the generated code samples (Format: [[code...], [code...]])")
    parser.add_argument("--eval_output_file", type=str, default=None, 
                        help="Optional: Custom path to save the evaluation metrics")
    
    # 情感 Trigger 参数
    parser.add_argument("--sentiment_trigger", type=str, default=None,
                        help="Trigger phrase to inject sentiment from JSON")
                    
    parser.add_argument("--emotion_json", type=str, default=None, 
                        help="Path to the developer emotions context JSON file")

    parser.add_argument("--use_cot", type=str, default="false", choices=["true", "false"],
                    help="Whether to use Chain-of-Thought (true/false). Default is false.")

    parser.add_argument("--use_fewshot", type=str, default="false", choices=["true", "false"],
                    help="Whether to use Few-Shot prompting (true/false). Default is false.")
    
    args = parser.parse_args()

    # 1. 加载标准化数据
    print(f"Loading dataset: {args.dataset}...")
    # 注意：APPS 数据集加载可能较慢
    problems, task_ids = load_standardized_dataset(args.dataset)
    print(f"Loaded {len(task_ids)} tasks.")
    
    # 2. 加载模型
    print(f"Loading model: {args.model}...")
    if args.sentiment_trigger:
        print(f"Applied Sentiment Trigger: {args.sentiment_trigger}")
        
    llm = load_model(args.model, sentiment_trigger=args.sentiment_trigger, json_path=args.emotion_json)

    # 3. 识别任务类型 (决定 Prompt 包装策略)
    is_mbpp_task = "mbpp" in args.dataset
    is_apps_task = "apps" in args.dataset
    
    # MBPP 和 APPS 通常是自然语言描述，适合用 Chat 模板处理
    # HumanEval 通常是代码补全，适合直接输入 Prompt
    use_chat_format = is_mbpp_task or is_apps_task

    # 4. 批量生成
    print(f"Generating {args.n_samples} samples per prompt...")
    
    all_predictions = []  # List[List[str]] -> 最终保存的格式
    all_references = []   # List[str] -> 用于评估的测试代码

    use_cot_bool = args.use_cot.lower() == "true"
    use_fewshot_bool = args.use_fewshot.lower() == "true"

    for task_id in tqdm(task_ids):
        problem = problems[task_id]
        prompt = problem["prompt"]
        
        # 生成代码
        # batch_codes: List[str], 长度为 n_samples
        batch_codes = llm.generate([prompt], n_samples=args.n_samples, is_mbpp=use_chat_format, 
                                    use_cot=use_cot_bool, use_fewshot=use_fewshot_bool, 
                                    dataset_name=args.dataset,
                                    max_new_tokens=1024 if use_cot_bool else 512)[0]
        
        # 数据后处理：确保代码完整可执行
        final_candidates = []
        for code in batch_codes:
            if not use_chat_format: 
                # === HumanEval 专用智能拼接逻辑 (修复版) ===
                cleaned_code = code.strip()
                cleaned_prompt = prompt.strip()
                
                # 提取 Prompt 中的 Import 语句 (这是 DeepSeek/Qwen 变差的关键补丁)
                # 也就是无论模型怎么写，我们都先把 imports 拿出来备用
                prompt_lines = cleaned_prompt.split('\n')
                import_lines = [line for line in prompt_lines if line.startswith("import ") or line.startswith("from ")]
                import_header = "\n".join(import_lines) + "\n" if import_lines else ""

                # 1. 完美情况：模型把 Prompt 完整抄了一遍
                if cleaned_prompt in cleaned_code:
                    final_candidates.append(cleaned_code)
                
                # 2. 模型重写了函数 (DeepSeek/Qwen/Llama3 常见行为)
                # 它们倾向于输出 "def func(): ..." 包含了函数头，但往往漏掉了 import
                elif "def " in cleaned_code:
                    # 如果生成的代码里没有 import，手动强行加上
                    if import_header.strip() and import_header.strip() not in cleaned_code:
                        final_candidates.append(import_header + cleaned_code)
                    else:
                        final_candidates.append(cleaned_code)
                
                # 3. 其他情况 (CodeLlama 常见行为)
                # 模型只写了 body (缩进的代码)，需要拼接 Prompt
                else:
                    final_candidates.append(prompt + code)
            else:
                # MBPP / APPS 逻辑保持不变
                if args.dataset == "apps":
                    cleaned_code = code
                    if "```python" in code:
                        cleaned_code = code.split("```python")[1].split("```")[0]
                    elif "```" in code:
                        cleaned_code = code.split("```")[1].split("```")[0]
                    final_candidates.append(cleaned_code.strip())
                else:
                    final_candidates.append(code)

        # 收集结果
        all_predictions.append(final_candidates)
        all_references.append(problem["test_code"])
        
        # === ✅ 新增：手动清理显存 ===
        # 1. 强制 Python 进行垃圾回收，清理不再使用的变量
        gc.collect()
        
        # 2. 强制 PyTorch 清空 CUDA 缓存
        torch.cuda.empty_cache()
        
        # (可选) 如果显存碎片化非常严重，可以同步一下
        # torch.cuda.synchronize()

    # 5. 调用 CodeEval 评估
    print("Running evaluation (Executing code)...")
    
    # 动态计算需要评估的 k 值
    k_list = [1, 5, 10]
    k_list = [k for k in k_list if k <= args.n_samples]

    if not k_list:
        k_list = [1] # 兜底，防止 n_samples < 1

    pass_at_k, detailed_results = compute_code_eval(
        predictions=all_predictions,
        references=all_references,
        k=k_list,
        num_workers=4, # 根据服务器 CPU 核心数调整
        timeout=3.0    # 每个测试用例的超时时间 (秒)
    )

    print("\n" + "="*35)
    print(f"📊 Evaluation Results: {args.dataset}")
    if args.sentiment_trigger:
        print(f"🧩 Trigger: {args.sentiment_trigger}")
    print(pass_at_k)
    print("="*35 + "\n")

    # 6. 保存结果
    
    # (A) 保存生成样本
    # 格式: [[sample1_1, sample1_2...], [sample2_1, sample2_2...]]
    save_json(all_predictions, args.output_file)
    
    # (B) 保存评估指标
    if args.eval_output_file:
        final_eval_path = args.eval_output_file
    else:
        # 默认命名规则
        final_eval_path = args.output_file.replace(".json", "_eval.json")

    end_time = datetime.now()
    duration = end_time - start_time
    
    print("-" * 50)
    print(f"✅ [FINISHED] Process ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ [DURATION] Total time taken: {duration}")
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