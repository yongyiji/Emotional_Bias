import torch
import json
import os
import gc  
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModel
from tqdm import tqdm


DATASET_FEW_SHOTS = {
    # === HumanEval 3-Shot ===
    # 特点：Input 是 "Function Header + Docstring", Output 是 "Function Body"
    "humaneval": [
        {
            "role": "user",
            "content": "You are a Python expert. Please implement the following function. Ensure the code is complete and self-contained.\n\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
        },
        {
            "role": "assistant",
            "content": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False"
        },
        {
            "role": "user",
            "content": "You are a Python expert. Please implement the following function. Ensure the code is complete and self-contained.\n\nfrom typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"
        },
        {
            "role": "assistant",
            "content": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return result"
        },
        {
            "role": "user",
            "content": "You are a Python expert. Please implement the following function. Ensure the code is complete and self-contained.\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n"
        },
        {
            "role": "assistant",
            "content": "def truncate_number(number: float) -> float:\n    return number % 1.0"
        }
    ],

    # === HumanEval+ 3-Shot ===
    # 通常可以用 HumanEval 的，或者稍微加强一点的例子
    "humaneval+": [
        {
            "role": "user",
            "content": "You are a Python expert. Please implement the following function. Ensure the code is complete and self-contained.\n\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
        },
        {
            "role": "assistant",
            "content": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False"
        },
        {
            "role": "user",
            "content": "You are a Python expert. Please implement the following function. Ensure the code is complete and self-contained.\n\nfrom typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"
        },
        {
            "role": "assistant",
            "content": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return result"
        },
        {
            "role": "user",
            "content": "You are a Python expert. Please implement the following function. Ensure the code is complete and self-contained.\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n"
        },
        {
            "role": "assistant",
            "content": "def truncate_number(number: float) -> float:\n    return number % 1.0"
        }
    ],

    # === MBPP 3-Shot ===
    # 特点：Input 是自然语言指令，Output 是完整函数
    "mbpp": [
        {
            "role": "user",
            "content": "Write a function to sort a given matrix in ascending order according to the sum of its rows."
        },
        {
            "role": "assistant",
            "content": "def sort_matrix(M):\r\n    result = sorted(M, key=sum)\r\n    return result"
        },
        {
            "role": "user",
            "content": "rite a function to count the most common words in a dictionary."
        },
        {
            "role": "assistant",
            "content": "from collections import Counter\r\ndef count_common(words):\r\n  word_counts = Counter(words)\r\n  top_four = word_counts.most_common(4)\r\n  return (top_four)\r\n"
        },
        {
            "role": "user",
            "content": "Write a python function to find the volume of a triangular prism."
        },
        {
            "role": "assistant",
            "content": "def find_Volume(l,b,h) : \r\n    return ((l * b * h) / 2)"
        }
    ],

    # === MBPP+ 3-Shot ===
    "mbpp+": [
        {
            "role": "user",
            "content": "Write a function to sort a given matrix in ascending order according to the sum of its rows."
        },
        {
            "role": "assistant",
            "content": "def sort_matrix(M):\r\n    result = sorted(M, key=sum)\r\n    return result"
        },
        {
            "role": "user",
            "content": "rite a function to count the most common words in a dictionary."
        },
        {
            "role": "assistant",
            "content": "from collections import Counter\r\ndef count_common(words):\r\n  word_counts = Counter(words)\r\n  top_four = word_counts.most_common(4)\r\n  return (top_four)\r\n"
        },
        {
            "role": "user",
            "content": "Write a python function to find the volume of a triangular prism."
        },
        {
            "role": "assistant",
            "content": "def find_Volume(l,b,h) : \r\n    return ((l * b * h) / 2)"
        }
    ]
}



class GenericLLM(BaseModel):
    def __init__(self, model_path, device="cuda", sentiment_trigger=None, json_path=None):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 保存 trigger 供 generate 使用
        self.sentiment_trigger = sentiment_trigger
        
        # 初始化情绪上下文变量
        self.context_instruction = None
        self.context_response = None

        # 基础 System Prompt
        base_system = "You are a developer. You need to answer the following coding problem and output only the code."

        # === 加载包含对话历史的 JSON ===
        # json_path = "/home/y/yj171/Sentiment_bias/sentiment_conversation_prompt/developer_emotions_context.json"

        if sentiment_trigger:
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    found = False
                    for item in data:
                        if item.get("trigger") == sentiment_trigger:
                            self.context_instruction = item.get("context_instruction")
                            self.context_response = item.get("context_response")
                            print(f"✅ Loaded emotional context for trigger: '{sentiment_trigger}'")
                            found = True
                            break
                    
                    if not found:
                        print(f"⚠️ Trigger '{sentiment_trigger}' not found in JSON.")
                except Exception as e:
                    print(f"❌ Error loading JSON: {e}.")
            else:
                print(f"❌ JSON file not found at {json_path}.")

        self.system_prompt = base_system
        print(f"📋 System Prompt: {self.system_prompt}") 
        if self.context_instruction:
            print(f"📚 Context Loaded: Will inject emotional history.")

    def generate(self, prompts, n_samples=1, max_new_tokens=512, is_mbpp=False, use_cot=False, use_fewshot=False, dataset_name=None):
        all_outputs = []
        BATCH_SIZE_PER_PASS = 10 
        
        try:
            model_name_lower = self.model.config._name_or_path.lower()
        except Exception:
            model_name_lower = ""

        # 判断是否为 StarCoder
        is_starcoder = "starcoder" in model_name_lower
        disable_system_role = (
            "starcoder" in model_name_lower or 
            "codegemma" in model_name_lower or
            "gemma" in model_name_lower or
            "codegen" in model_name_lower or
            "codestral" in model_name_lower or
            "mistral" in model_name_lower
        )

        selected_few_shots = []
        if use_fewshot and dataset_name in DATASET_FEW_SHOTS:
            selected_few_shots = DATASET_FEW_SHOTS[dataset_name]
            # print(f"🔎 Using 3-shot examples for dataset: {dataset_name}")
        elif use_fewshot:
            # 用户开了 fewshot 但是数据集名字不对，这里选择不加载 default，直接打印警告
            print(f"⚠️ Warning: Few-shot requested but no examples found for '{dataset_name}'. Proceeding with 0-shot.")
            selected_few_shots = []

        for prompt in prompts:
            if use_cot:
                # 将 CoT 咒语直接拼接到原始题目后面
                prompt = prompt + "\n\nPlease think step by step and then provide the code."


            input_text = prompt
            
            is_codellama_instruct = "codellama" in model_name_lower and "instruct" in model_name_lower
            
            # === 构造 Chat Messages ===
            messages = []
            
            # (A) System Prompt 逻辑控制
            # 只有当不是 StarCoder 时，才添加 System Role
            # StarCoder 直接跳过这一步 (即放弃 System Prompt)
            if self.system_prompt and not disable_system_role:
                messages.append({"role": "system", "content": self.system_prompt})
            
            # (B) [修改] 注入 Few-Shot Examples
            # 只有当 selected_few_shots 非空时才注入
            if selected_few_shots:
                messages.extend(selected_few_shots)


            # (B) 注入情绪历史 (Fake History)
            if self.context_instruction and self.context_response:
                messages.append({"role": "user", "content": self.context_instruction})
                messages.append({"role": "assistant", "content": self.context_response})
            
            # (C) 当前用户的真实提问 (代码题)
            messages.append({"role": "user", "content": prompt})

            # --- 分支处理：CodeLlama vs 普通 Chat Template ---
            
            if is_codellama_instruct:
                # CodeLlama 逻辑保持不变 (手动拼接)
                sys_str = f"<<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n" if self.system_prompt else ""
                
                if self.context_instruction and self.context_response:
                    input_text = (
                        f"<s>[INST] {sys_str}{self.context_instruction} [/INST] "
                        f"{self.context_response} </s>"
                        f"<s>[INST] {prompt} [/INST]"
                    )
                else:
                    input_text = f"<s>[INST] {sys_str}{prompt} [/INST]"
                    
            elif self.tokenizer.chat_template:
                # 使用 chat_template 处理 messages
                try:
                    input_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"Template apply failed: {e}, falling back.")
                    # 极端兜底：如果还报错，只发 Prompt
                    input_text = prompt
            # >>>>>>>> 在这里添加打印代码 <<<<<<<<
            # print("\n" + "="*20 + " DEBUG: PROMPT CHECK " + "="*20)
            # print(f"【Current Model】: {self.model.config._name_or_path}")
            # print(f"【Emotional Trigger】: {self.sentiment_trigger}")
            # print("-" * 10 + " Full Input Text " + "-" * 10)
            # print(input_text)  # <--- 核心：打印最终喂给模型的文本
            # print("="*60 + "\n")
            # >>>>>>>> 添加结束 <<<<<<<<

            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            input_len = len(inputs.input_ids[0])
            
            prompt_outputs = []
            samples_generated_so_far = 0
            
            while samples_generated_so_far < n_samples:
                remaining = n_samples - samples_generated_so_far
                current_batch_size = min(remaining, BATCH_SIZE_PER_PASS)
                
                with torch.no_grad():
                    try:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True if n_samples > 1 else False,
                            temperature=0.2 if n_samples > 1 else 0.0,
                            top_p=0.95,
                            num_return_sequences=current_batch_size, 
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        for i in range(len(generated_ids)):
                            output_ids = generated_ids[i][input_len:]
                            code = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                            
                            if "```python" in code:
                                code = code.split("```python")[1].split("```")[0]
                            elif "```" in code:
                                code = code.split("```")[1].split("```")[0]
                            elif "[PYTHON]" in code:
                                code = code.split("[PYTHON]")[1].split("[/PYTHON]")[0]
                            
                            prompt_outputs.append(code.strip())

                    except Exception as e:
                        print(f"❌ Error during generation batch: {e}")
                        prompt_outputs.extend([""] * current_batch_size)
                    
                    samples_generated_so_far += current_batch_size
                    
                    del generated_ids
                    gc.collect()
                    torch.cuda.empty_cache()

            all_outputs.append(prompt_outputs)
            
        return all_outputs