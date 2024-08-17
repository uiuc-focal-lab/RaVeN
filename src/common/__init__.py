import re
from enum import Enum




# Domains used for verification
class Domain(Enum):
    DEEPZ = 1
    DEEPPOLY = 2
    INDIVIDUAL = 3
    IOFORMULATION = 4
    RAVEN = 5
    LIRPA = 6


# Used for status of the verifier (currently unused)
class Status(Enum):
    VERIFIED = 1
    ADV_EXAMPLE = 2
    UNKNOWN = 3
    MISS_CLASSIFIED = 4
