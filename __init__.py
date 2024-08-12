from .run import NeuralNeighbor

NODE_CLASS_MAPPINGS = {
    "NeuralNeighbor": NeuralNeighbor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralNeighbor": "Neural Neighbor",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
