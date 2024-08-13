from .run import NeuralNeighbor, CAST, EFDM, MicroAST

NODE_CLASS_MAPPINGS = {
    "NeuralNeighbor": NeuralNeighbor,
    "CAST": CAST,
    "EFDM": EFDM,
    "MicroAST": MicroAST,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralNeighbor": "Neural Neighbor",
    "CAST": "CAST",
    "EFDM": "EFDM",
    "MicroAST": "MicroAST",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
