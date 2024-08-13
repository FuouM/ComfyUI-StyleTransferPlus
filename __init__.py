from .run import NeuralNeighbor, CAST, EFDM

NODE_CLASS_MAPPINGS = {
    "NeuralNeighbor": NeuralNeighbor,
    "CAST": CAST,
    "EFDM": EFDM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralNeighbor": "Neural Neighbor",
    "CAST": "CAST",
    "EFDM": "EFDM",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
