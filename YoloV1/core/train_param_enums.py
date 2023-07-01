from enum import Enum, unique, auto

@unique
class LoadStrategy(Enum):
    NONE = 0
    MODEL = auto()
    MODEL_OPTIMIZER = auto()

@unique
class SaveStrategy(Enum):
    NONE = 0
    FREQUENCY = auto()
    BEST_MODEL = auto()

@unique
class MetricType(Enum):
    METRIC = 0
    LOSS = auto()

@unique
class TypeLoadModel(Enum):
    BEST_METRIC = 0
    BEST_LOSS = auto()