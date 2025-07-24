SEQUENCE_LENGTH = 77
NUM_RETURN_BUCKETS = 128
DEFAULT_VOCAB_SIZE = 32

FEN_CHARACTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'p', 'n', 'r', 'k', 'q', 'P', 'B', 'N', 'R', 'Q', 'K',
    'w', '.', '/'
]

CHARACTERS_INDEX = {char: idx for idx, char in enumerate(FEN_CHARACTERS)}
SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})

MODEL_SIZES = {
    "small": {"params": "9M", "embedding_dim": 256, "num_heads": 8, "num_layers": 8},
    "medium": {"params": "136M", "embedding_dim": 1024, "num_heads": 8, "num_layers": 8},
    "large": {"params": "270M", "embedding_dim": 1024, "num_heads": 8, "num_layers": 16},
}