"""
Core GNN module with model registry.

This module provides a unified interface for accessing all GNN models
via the get_model() factory function.
"""

from core.GCN_scratch.model import GCN
from core.GAT_scratch.model import GAT
from core.GIN_scratch.model import GIN
from core.GraphSAGE_scratch.model import GraphSAGE

MODEL_REGISTRY = {
    "GCN": GCN,
    "GAT": GAT,
    "GIN": GIN,
    "GraphSAGE": GraphSAGE,
}


def get_model(name, *args, **kwargs):
    """
    Factory function to get a GNN model by name.
    
    Args:
        name: Model name, one of: "GCN", "GAT", "GIN", "GraphSAGE"
        *args: Positional arguments passed to model constructor
        **kwargs: Keyword arguments passed to model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        KeyError: If model name is not in registry
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](*args, **kwargs)
