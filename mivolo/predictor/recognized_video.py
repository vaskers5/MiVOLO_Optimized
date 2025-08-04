from dataclasses import dataclass
from typing import Any, Dict, List

from mivolo.structures import AGE_GENDER_TYPE

from .detected_person import DetectedPerson


@dataclass
class RecognizedVideo:
    detected_persons: list[List[DetectedPerson]]
    detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_persons": [[person.to_dict() for person in frame] for frame in self.detected_persons],
            "detected_objects_history": self.detected_objects_history,
        }
