from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    model_name: str
    normalize: bool = True 

"""
Defines EmbeddingConfig dataclass with model_name field. Controls which sentence transformer model is used for embedding.
"""