from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .gender_enum import GenderEnum


@dataclass
class DetectedPerson:
    age: int
    gender: GenderEnum
    face_bbox: Optional[np.ndarray] = None
    face_confidence: Optional[float] = None
    body_bbox: Optional[np.ndarray] = None
    body_confidence: Optional[float] = None
    vit_confidence: Optional[float] = None

    def has_face(self) -> bool:
        """Check if face detection is available."""
        return self.face_bbox is not None

    def has_body(self) -> bool:
        """Check if body detection is available."""
        return self.body_bbox is not None

    def is_complete_detection(self) -> bool:
        """Check if both face and body are detected."""
        return self.has_face() and self.has_body()

    def get_face_area(self) -> Optional[float]:
        """Calculate face bounding box area."""
        if self.face_bbox is None:
            return None
        return float((self.face_bbox[2] - self.face_bbox[0]) * (self.face_bbox[3] - self.face_bbox[1]))

    def get_body_area(self) -> Optional[float]:
        """Calculate body bounding box area."""
        if self.body_bbox is None:
            return None
        return float((self.body_bbox[2] - self.body_bbox[0]) * (self.body_bbox[3] - self.body_bbox[1]))

    def get_face_center(self) -> Optional[Tuple[float, float]]:
        """Get face bounding box center coordinates."""
        if self.face_bbox is None:
            return None
        return (
            float((self.face_bbox[0] + self.face_bbox[2]) / 2),
            float((self.face_bbox[1] + self.face_bbox[3]) / 2)
        )

    def get_body_center(self) -> Optional[Tuple[float, float]]:
        """Get body bounding box center coordinates."""
        if self.body_bbox is None:
            return None
        return (
            float((self.body_bbox[0] + self.body_bbox[2]) / 2),
            float((self.body_bbox[1] + self.body_bbox[3]) / 2)
        )

    def get_average_confidence(self) -> float:
        """Calculate average confidence from all available confidence scores."""
        confidences = [c for c in [self.face_confidence, self.body_confidence, self.vit_confidence] if c is not None]
        return float(np.mean(confidences)) if confidences else 0.0

    def get_quality_score(self) -> str:
        """Get a quality assessment based on confidence scores."""
        avg_conf = self.get_average_confidence()
        if avg_conf > 0.9:
            return "High"
        elif avg_conf > 0.7:
            return "Medium"
        elif avg_conf > 0.5:
            return "Low"
        else:
            return "Very Low"

    def to_dict(self) -> dict[str, Any]:
        """Convert DetectedPerson to dictionary representation."""
        return {
            "age": self.age,
            "gender": self.gender.value,
            "face_bbox": self.face_bbox.tolist() if self.face_bbox is not None else None,
            "face_confidence": self.face_confidence,
            "body_bbox": self.body_bbox.tolist() if self.body_bbox is not None else None,
            "body_confidence": self.body_confidence,
            "vit_confidence": self.vit_confidence,
            "has_face": self.has_face(),
            "has_body": self.has_body(),
            "is_complete": self.is_complete_detection(),
            "face_area": self.get_face_area(),
            "body_area": self.get_body_area(),
            "face_center": self.get_face_center(),
            "body_center": self.get_body_center(),
            "average_confidence": self.get_average_confidence(),
            "quality": self.get_quality_score()
        }
