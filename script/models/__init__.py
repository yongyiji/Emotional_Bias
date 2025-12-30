from .generic_llm_context import GenericLLM


def load_model(model_path, device="cuda", sentiment_trigger=None, json_path=None):
    return GenericLLM(model_path, device=device, sentiment_trigger=sentiment_trigger, json_path=json_path)
