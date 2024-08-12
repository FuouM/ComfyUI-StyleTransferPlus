from .run import NeuralNeighbor, CAST

NODE_CLASS_MAPPINGS = {
    "NeuralNeighbor": NeuralNeighbor,
    "CAST": CAST,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralNeighbor": "Neural Neighbor",
    "CAST": "CAST",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
