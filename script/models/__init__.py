from .generic_llm_context import GenericLLM

# 修改 load_model 以接收 sentiment_trigger
def load_model(model_path, device="cuda", sentiment_trigger=None, json_path=None):
    # 这里我们假设使用 GenericLLM
    # 如果你有其他模型类，也需要相应处理
    return GenericLLM(model_path, device=device, sentiment_trigger=sentiment_trigger, json_path=json_path)