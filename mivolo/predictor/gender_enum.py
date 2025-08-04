from enum import Enum


class GenderEnum(str, Enum):
    """Gender enumeration compatible with older Python versions"""
    MALE = "male"
    FEMALE = "female"
