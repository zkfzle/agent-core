from enum import Enum
from dataclasses import dataclass


@dataclass
class ExtractedDataType(Enum):
    USER = "user"


@dataclass
class ExtractedData(object):
    type: ExtractedDataType
    key: str
    value: str
