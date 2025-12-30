import os
import json
import multiprocessing
import concurrent.futures
import numpy as np
import contextlib
import signal
import itertools
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from datasets import load_dataset
import re

# 1: Execution Logic


class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def unsafe_execute(test_program, timeout):
    with create_tempdir():
        try:
            exec_globals = {}
            with time_limit(timeout):
                if "rm -rf" in test_program or "os.system" in test_program: 
                    return "failed: unsafe code detected"
                
                exec(test_program, exec_globals)
            return "passed"
        except TimeoutException:
            return "timed out"
        except AssertionError as e:
            return f"failed: assertion error {str(e)}"
        except Exception as e:
            return f"failed: {e}"


def _unsafe_execute_wrapper(program, result_list, timeout):
    from io import StringIO
    import contextlib

    capture_out = StringIO()
    try:
        exec_globals = {"__builtins__": __builtins__}
        
        with contextlib.redirect_stdout(capture_out):
            exec(program, exec_globals)
        
        result_list.append("passed")
            
    except Exception as e:
        result_list.append(f"failed: {str(e)}")
    finally:
        capture_out.close()

def check_correctness(test_program, timeout, task_id, completion_id):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=_unsafe_execute_wrapper, args=(test_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    
    if p.is_alive():
        p.terminate()
        p.join()
        status = "timed out"
    else:
        if not result:
            status = "failed: process died"
        else:
            status = result[0]

    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "passed": status == "passed",
        "result": status,
        "completion": "" 
    }

@contextlib.contextmanager
def create_tempdir():
    yield


# Part 2: APPS Test Generator


def generate_apps_test_code(input_output):
    try:
        if isinstance(input_output, str):
            io_data = json.loads(input_output)
        else:
            io_data = input_output
    except:
        return "raise ValueError('Input/Output parsing failed')"

    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    
    if not inputs:
        return "raise ValueError('No test cases found')"

    test_harness = r"""
import sys
import io
import math

def _check_apps_solution():
    inputs = """ + str(inputs) + r"""
    outputs = """ + str(outputs) + r"""
    
    if 'solve' not in globals():
        raise NotImplementedError("Solution must be wrapped in 'def solve():'")

    for i, (inp, exp) in enumerate(zip(inputs, outputs)):
        f_in = io.StringIO(inp)
        f_out = io.StringIO()
        original_stdin, original_stdout = sys.stdin, sys.stdout
        
        try:
            sys.stdin, sys.stdout = f_in, f_out
            solve()
        finally:
            sys.stdin, sys.stdout = original_stdin, original_stdout
        
        result = f_out.getvalue().strip()
        expected = str(exp).strip()
        

        if result != expected:
            raise AssertionError(f"Test {i} Failed. Exp: {expected}, Got: {result}")

_check_apps_solution()
"""
    return test_harness


# Part 3: Dataset Loading & Standardization

def load_standardized_dataset(dataset_name):
    """
    Returns:
        problems (dict): {task_id: {prompt, test_code, entry_point, ...}}
        task_ids (list): list of task_ids keys
    """
    problems = {}
    

    if dataset_name == "humaneval":
        from human_eval.data import read_problems
        raw_data = read_problems()
    elif dataset_name == "humaneval+":
        # EvalPlus 的 HumanEval+
        raw_data = get_human_eval_plus()
        
    elif dataset_name == "mbpp":
        print("Loading original MBPP (sanitized) from Hugging Face...")
        
        ds = load_dataset("mbpp", "sanitized", split="train+validation+test+prompt")
        
        raw_data = {}
        for row in ds:
            prompt_content = row.get('prompt') or row.get('text')
            if prompt_content is None:
                raise ValueError(f"Cannot find prompt column. Available keys: {list(row.keys())}")

            entry_point = ""
            if "code" in row:
                match = re.search(r"def\s+(\w+)\s*\(", row["code"])
                if match:
                    entry_point = match.group(1)
            
            raw_data[row['task_id']] = {
                "prompt": prompt_content,
                "test_list": row['test_list'], 
                "entry_point": entry_point
            }
    elif dataset_name == "mbpp+":
        raw_data = get_mbpp_plus()

    elif dataset_name == "apps":
        print("Loading APPS dataset (codeparrot/apps)...")
        try:
            ds = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
            
            print(f"Original test set size: {len(ds)}")

            ds = ds.filter(lambda example: example['difficulty'] == 'interview')
            print(f"Interview-level size: {len(ds)}")

            if len(ds) > 500:
                ds = ds.shuffle(seed=42).select(range(10))
            
            print(f"Final sampled size: {len(ds)}")

        except Exception as e:
            print(f"Error loading APPS: {e}")
            raise e

        for item in ds:
            task_id = f"APPS/{item['problem_id']}"
            
            prompt_text = item['question'] + "\n\n" + \
                          "IMPORTANT: Write your solution inside a function named `def solve():`.\n" + \
                          "Do not call the function yourself, just define it."
            
            test_code = generate_apps_test_code(item['input_output'])
            
            problems[task_id] = {
                "prompt": prompt_text,
                "test_code": test_code,
                "entry_point": "solve"
            }
        return problems, list(problems.keys())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    for task_id, item in raw_data.items():
        if "humaneval" in dataset_name:
            # HumanEval: Prompt + Completion + Test + Check
            test_code = item["test"] + f"\ncheck({item['entry_point']})"
            problems[task_id] = {
                "prompt": item["prompt"],
                "test_code": test_code,
                "entry_point": item["entry_point"]
            }
        elif "mbpp" in dataset_name:
            # MBPP: Header + Completion + Test Harness
            header = "import math\nfrom typing import List, Tuple, Dict, Any, Optional\n\n"
            
            if 'test_list' in item:
                tests = "\n".join(item['test_list'])
            elif 'test' in item:
                tests = item['test']
            elif 'assertion' in item:
                tests = item['assertion']
            else:
                print(f"Warning: No test found for task {task_id}. Keys: {item.keys()}")
                tests = ""

            if dataset_name == "mbpp+" and 'test' in item and item['test']:
                final_test_code = header + item['test']
            else:
                final_test_code = header + tests
            
            problems[task_id] = {
                "prompt": item["prompt"],
                "test_code": final_test_code,
                "entry_point": item["entry_point"]
            }
    
    return problems, list(problems.keys())


# Part 4: Pass@k Calculation & Main Compute

_WARNING = """
################################################################################
                            !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
"""

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def compute_code_eval(predictions, references, k=[1, 10, 100], num_workers=4, timeout=3.0):
    """
    Main entry point provided by user.
    """
    if os.getenv("HF_ALLOW_CODE_EVAL", "0") != "1":
        raise ValueError(_WARNING)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
            for candidate in candidates:
                test_program = candidate + "\n" + test_case
                args = (test_program, timeout, task_id, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    if not isinstance(ks, (list, tuple)):
        ks = [ks]
    
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

    return pass_at_k, results
