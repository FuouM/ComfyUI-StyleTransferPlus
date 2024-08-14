from .run import (
    AesFA,
    AesFAStyleBlend,
    CAST,
    EFDM,
    TSSAT,
    AesPA,
    MicroAST,
    NeuralNeighbor,
    UniST,
    UniST_Video,
)
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
    "TSSAT": TSSAT,
    "AESFA": AesFA,
    "AesFAStyleBlend": AesFAStyleBlend,
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
    "TSSAT": "TSSAT",
    "AESFA": "AESFA",
    "AesFAStyleBlend": "AesFA Styles Blending",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
