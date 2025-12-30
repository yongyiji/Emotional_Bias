from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompts, n_samples=1, use_cot=False, use_fewshot=False, dataset_name=None):
        pass