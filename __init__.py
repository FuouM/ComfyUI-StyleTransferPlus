from .run import NeuralNeighbor, CAST, EFDM, MicroAST, UniST, UniST_Video, AesPA
from .run_extra import CoralColorTransfer

NODE_CLASS_MAPPINGS = {
    "NeuralNeighbor": NeuralNeighbor,
    "CAST": CAST,
    "EFDM": EFDM,
    "MicroAST": MicroAST,
    "CoralColorTransfer": CoralColorTransfer,
    "UniST": UniST,
    "UniST_Video": UniST_Video,
    "AesPA": AesPA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralNeighbor": "Neural Neighbor",
    "CAST": "CAST",
    "EFDM": "EFDM",
    "MicroAST": "MicroAST",
    "CoralColorTransfer": "Coral Color Transfer",
    "UniST": "UniST",
    "UniST_Video": "UniST Video",
    "AesPA": "AesPA-Net",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
