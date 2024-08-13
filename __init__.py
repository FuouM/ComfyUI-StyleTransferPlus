from .run import NeuralNeighbor, CAST, EFDM, MicroAST
from .run_extra import CoralColorTransfer

NODE_CLASS_MAPPINGS = {
    "NeuralNeighbor": NeuralNeighbor,
    "CAST": CAST,
    "EFDM": EFDM,
    "MicroAST": MicroAST,
    "CoralColorTransfer": CoralColorTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralNeighbor": "Neural Neighbor",
    "CAST": "CAST",
    "EFDM": "EFDM",
    "MicroAST": "MicroAST",
    "CoralColorTransfer": "Coral Color Transfer",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
