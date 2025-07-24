class ChessLLMError(Exception):
    pass


class DataError(ChessLLMError):
    pass


class ModelError(ChessLLMError):
    pass


class TrainingError(ChessLLMError):
    pass


class ConfigurationError(ChessLLMError):
    pass


class CheckpointError(ChessLLMError):
    pass