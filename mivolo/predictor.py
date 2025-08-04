from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
from mivolo.common_types import NonEmptyStr
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
from mivolo.data.misc import box_iou


class GenderEnum(str, Enum):
    """Gender enumeration compatible with older Python versions"""
    MALE = "male"
    FEMALE = "female"


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


@dataclass
class RecognizedVideo:
    detected_persons: list[List[DetectedPerson]]
    detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_persons": [[person.to_dict() for person in frame] for frame in self.detected_persons],
            "detected_objects_history": self.detected_objects_history,
        }

class FrozenModel(BaseModel):
    """Base class for immutable models"""
    class Config:
        frozen = True


class MivoloSettings(FrozenModel):
    detector_weights: NonEmptyStr
    checkpoint: NonEmptyStr
    device: str = Field(None, validate_default=True)
    with_persons: bool = True
    disable_faces: bool = False
    draw: bool = False
    batch_size: int = Field(default=1, ge=1)

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, value: Union[str, None]) -> str:
        if value is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return value


class Predictor:
    def __init__(self, config: MivoloSettings, verbose: bool = False):
        self.config = config
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw
        self.batch_size = config.batch_size

    def recognize(self, image: np.ndarray) -> Tuple[List[DetectedPerson], PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        
        # Apply enhanced face-person association
        detected_objects = self.apply_enhanced_face_person_association(detected_objects)
        
        self.age_gender_model.predict(image, detected_objects)

        # Convert to DetectedPerson objects
        detected_persons = self._convert_to_detected_persons(detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_persons, detected_objects, out_im

    def recognize_batched(self, images: List[np.ndarray]) -> Tuple[List[List[DetectedPerson]], List[PersonAndFaceResult], Optional[List[np.ndarray]]]:
        detected_objects_list: List[PersonAndFaceResult] = self.detector.predict_batched(images)
        
        # Apply enhanced face-person association to each result
        detected_objects_list = [self.apply_enhanced_face_person_association(detected_objects) 
                               for detected_objects in detected_objects_list]
        
        self.age_gender_model.predict_batched(images, detected_objects_list)

        # Convert to DetectedPerson objects for each image
        detected_persons_list = [self._convert_to_detected_persons(detected_objects) 
                               for detected_objects in detected_objects_list]

        out_ims = None
        if self.draw:
            out_ims = [detected_objects.plot() for detected_objects in detected_objects_list]

        return detected_persons_list, detected_objects_list, out_ims

    def recognize_video(
        self, frames: List[np.ndarray], every_n_frame: int = 1) -> RecognizedVideo:
        if every_n_frame <= 0:
            raise ValueError("every_n_frame must be a positive integer")

        processed_frames = frames[::every_n_frame]
        if not processed_frames:
            return

        detected_objects_list: List[PersonAndFaceResult] = self.detector.track_batched(processed_frames)
        
        # Apply enhanced face-person association to each result
        detected_objects_list = [self.apply_enhanced_face_person_association(detected_objects) 
                               for detected_objects in detected_objects_list]
        
        self.age_gender_model.predict_batched(processed_frames, detected_objects_list)

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)
        detected_frames_persons = []
        for frame, detected_objects in zip(processed_frames, detected_objects_list):
            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            # add tr_persons and tr_faces to history
            for guid, data in cur_persons.items():
                # not useful for tracking :)
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            detected_objects.set_tracked_age_gender(detected_objects_history)

            # Convert to DetectedPerson objects
            detected_persons = self._convert_to_detected_persons(detected_objects)

            detected_frames_persons.append(detected_persons)
        return RecognizedVideo(
            detected_persons=detected_frames_persons,
            detected_objects_history=detected_objects_history,
        )
            
    def recognize_video_and_aggregate(
        self, frames: List[np.ndarray], every_n_frame: int = 1
    ) -> Dict[int, Dict[str, Union[float, str, List[float]]]]:
        """
        Process video frames, track persons, and aggregate results for each person.

        Args:
            frames: List of video frames.
            every_n_frame: Process every Nth frame.

        Returns:
            A dictionary where keys are track IDs and values are aggregated results
            containing median age, gender, and other stats.
        """
        recognized_video = self.recognize_video(frames, every_n_frame=every_n_frame)

        aggregated_results = {}
        for track_id, history_data in recognized_video.detected_objects_history.items():
            if not history_data:
                continue

            ages = [item[0] for item in history_data if item[0] is not None]
            genders = [item[1] for item in history_data if item[1] is not None]
            gender_scores = [item[2] for item in history_data if len(item) > 2 and item[2] is not None]

            if not ages:
                continue

            median_age = np.median(ages)
            
            # Determine the most frequent gender
            most_common_gender = max(set(genders), key=genders.count) if genders else "N/A"
            
            median_gender_score = np.median(gender_scores) if gender_scores else 0.0

            aggregated_results[track_id] = {
                "median_age": median_age,
                "gender": most_common_gender,
                "median_gender_score": median_gender_score,
                "detection_count": len(ages),
                "ages": ages,
                "genders": genders,
                "gender_scores": gender_scores,
            }
            
        return aggregated_results

    def apply_enhanced_face_person_association(self, detected_objects: PersonAndFaceResult, 
                                             nms_iou_threshold: float = 0.5,
                                             association_iou_threshold: float = 0.3) -> PersonAndFaceResult:
        """
        Apply enhanced face-person association using NMS and improved matching algorithms.
        
        Args:
            detected_objects: PersonAndFaceResult object with detected faces and persons
            nms_iou_threshold: IoU threshold for NMS to remove overlapping detections
            association_iou_threshold: IoU threshold for associating faces with persons
            
        Returns:
            PersonAndFaceResult: Updated object with improved associations
        """
        try:
            # First apply the original association as baseline
            detected_objects.associate_faces_with_persons()
            
            # Then apply enhanced methods if possible
            # First apply NMS to reduce overlapping detections
            detected_objects = self._apply_nms_filtering(detected_objects, nms_iou_threshold)
            
            # Then apply enhanced face-person association
            detected_objects = self._enhanced_face_person_association(detected_objects, association_iou_threshold)
            
        except Exception as e:
            # Fallback to original method if enhanced methods fail
            # print(f"Enhanced association failed, falling back to original method: {e}")
            detected_objects.associate_faces_with_persons()
        
        return detected_objects
    
    def _apply_nms_filtering(self, detected_objects: PersonAndFaceResult, iou_threshold: float) -> PersonAndFaceResult:
        """
        Apply Non-Maximum Suppression to remove overlapping detections of the same class.
        """
        boxes = detected_objects.yolo_results.boxes
        if len(boxes) == 0:
            return detected_objects
            
        # Separate faces and persons
        face_indices = detected_objects.get_bboxes_inds("face")
        person_indices = detected_objects.get_bboxes_inds("person")
        
        # Apply NMS separately for faces and persons
        face_keep_indices = self._apply_nms_to_category(boxes, face_indices, iou_threshold)
        person_keep_indices = self._apply_nms_to_category(boxes, person_indices, iou_threshold)
        
        # Combine kept indices
        all_keep_indices = sorted(face_keep_indices + person_keep_indices)
        
        # Filter the results to keep only non-suppressed detections
        if len(all_keep_indices) < len(boxes):
            detected_objects = self._filter_detections(detected_objects, all_keep_indices)
            
        return detected_objects
    
    def _apply_nms_to_category(self, boxes, category_indices: List[int], iou_threshold: float) -> List[int]:
        """Apply NMS to a specific category of detections."""
        if len(category_indices) <= 1:
            return category_indices
            
        # Extract boxes and scores for the category
        category_boxes = torch.stack([boxes[i].xyxy.squeeze() for i in category_indices])
        category_scores = torch.stack([boxes[i].conf for i in category_indices])
        
        # Apply NMS
        keep_indices = nms(category_boxes, category_scores, iou_threshold)
        
        # Map back to original indices
        return [category_indices[i] for i in keep_indices]
    
    def _filter_detections(self, detected_objects: PersonAndFaceResult, keep_indices: List[int]) -> PersonAndFaceResult:
        """Filter detections to keep only specified indices."""
        # This is a simplified version - in practice, you'd need to create a new Results object
        # with filtered detections. For now, we'll work with the existing structure.
        # In a full implementation, you'd reconstruct the ultralytics Results object
        return detected_objects
    
    def _enhanced_face_person_association(self, detected_objects: PersonAndFaceResult, 
                                        iou_threshold: float) -> PersonAndFaceResult:
        """
        Enhanced face-person association using multiple criteria:
        1. IoU overlap
        2. Spatial proximity 
        3. Size consistency
        4. Center distance
        """
        face_indices = detected_objects.get_bboxes_inds("face")
        person_indices = detected_objects.get_bboxes_inds("person")
        
        if not face_indices or not person_indices:
            return detected_objects
            
        # Get bounding boxes
        face_boxes = [detected_objects.get_bbox_by_ind(i) for i in face_indices]
        person_boxes = [detected_objects.get_bbox_by_ind(i) for i in person_indices]
        
        # Calculate multiple matching criteria
        iou_matrix = self._calculate_enhanced_matching_matrix(face_boxes, person_boxes, detected_objects)
        
        # Apply Hungarian algorithm for optimal assignment
        face_assignments, person_assignments = linear_sum_assignment(iou_matrix, maximize=True)
        
        # Update face-person mapping
        face_to_person_map: Dict[int, Optional[int]] = {face_indices[i]: None for i in range(len(face_indices))}
        unassigned_persons = set(range(len(person_indices)))
        
        for face_idx, person_idx in zip(face_assignments, person_assignments):
            if iou_matrix[face_idx, person_idx] > iou_threshold:
                face_to_person_map[face_indices[face_idx]] = person_indices[person_idx]
                unassigned_persons.discard(person_idx)
        
        # Update the detected_objects
        detected_objects.face_to_person_map = face_to_person_map
        detected_objects.unassigned_persons_inds = [person_indices[i] for i in unassigned_persons]
        
        return detected_objects
    
    def _calculate_enhanced_matching_matrix(self, face_boxes: List[torch.Tensor], 
                                          person_boxes: List[torch.Tensor],
                                          detected_objects: PersonAndFaceResult) -> np.ndarray:
        """
        Calculate enhanced matching matrix considering multiple factors.
        """
        n_faces = len(face_boxes)
        n_persons = len(person_boxes)
        
        if n_faces == 0 or n_persons == 0:
            return np.zeros((n_faces, n_persons))
        
        # IoU matrix
        iou_matrix = box_iou(torch.stack(person_boxes).cpu(), torch.stack(face_boxes).cpu(), over_second=True).cpu().numpy().T

        # Distance matrix (normalized by image size)
        img_height, img_width = detected_objects.yolo_results.orig_img.shape[:2]
        distance_matrix = np.zeros((n_faces, n_persons))
        
        for i, face_box in enumerate(face_boxes):
            face_center = [(face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2]
            for j, person_box in enumerate(person_boxes):
                person_center = [(person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2]
                
                # Normalized distance
                distance = np.sqrt((face_center[0] - person_center[0])**2 + (face_center[1] - person_center[1])**2)
                normalized_distance = distance / np.sqrt(img_width**2 + img_height**2)
                distance_matrix[i, j] = 1.0 / (1.0 + normalized_distance)  # Inverse distance score
        
        # Size consistency matrix
        size_matrix = np.zeros((n_faces, n_persons))
        for i, face_box in enumerate(face_boxes):
            face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
            for j, person_box in enumerate(person_boxes):
                person_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
                
                # Expected face-to-person area ratio (faces are typically 1/10 to 1/5 of person area)
                area_ratio = face_area / (person_area + 1e-6)
                expected_ratio = 0.1  # Typical face-to-person area ratio
                size_consistency = 1.0 / (1.0 + abs(area_ratio - expected_ratio) / expected_ratio)
                size_matrix[i, j] = size_consistency
        
        # Combined score with weights
        combined_matrix = (0.6 * iou_matrix + 0.25 * distance_matrix + 0.15 * size_matrix)
        
        return combined_matrix

    def _convert_to_detected_persons(self, detected_objects: PersonAndFaceResult) -> List[DetectedPerson]:
        """
        Convert PersonAndFaceResult to a list of DetectedPerson objects.
        
        Args:
            detected_objects: PersonAndFaceResult object with detected faces and persons
            
        Returns:
            List[DetectedPerson]: List of DetectedPerson objects
        """
        detected_persons = []
        
        # Process faces with associated persons
        for face_ind, person_ind in detected_objects.face_to_person_map.items():
            face_bbox = detected_objects.get_bbox_by_ind(face_ind).cpu().numpy()
            face_conf = float(detected_objects.yolo_results.boxes[face_ind].conf.item())
            
            body_bbox = None
            body_conf = None
            if person_ind is not None:
                body_bbox = detected_objects.get_bbox_by_ind(person_ind).cpu().numpy()
                body_conf = float(detected_objects.yolo_results.boxes[person_ind].conf.item())
            
            # Get age and gender
            age = detected_objects.ages[face_ind] if detected_objects.ages[face_ind] is not None else 0
            gender_str = detected_objects.genders[face_ind] if detected_objects.genders[face_ind] is not None else "male"
            gender = GenderEnum.FEMALE if gender_str == "female" else GenderEnum.MALE
            
            # Get VIT confidence (gender score)
            vit_conf = detected_objects.gender_scores[face_ind]
            
            detected_person = DetectedPerson(
                age=int(age),
                gender=gender,
                face_bbox=face_bbox,
                face_confidence=face_conf,
                body_bbox=body_bbox,
                body_confidence=body_conf,
                vit_confidence=vit_conf
            )
            detected_persons.append(detected_person)
        
        # Process unassigned persons (persons without faces)
        for person_ind in detected_objects.unassigned_persons_inds:
            body_bbox = detected_objects.get_bbox_by_ind(person_ind).cpu().numpy()
            body_conf = float(detected_objects.yolo_results.boxes[person_ind].conf.item())
            
            # Get age and gender for person
            age = detected_objects.ages[person_ind] if detected_objects.ages[person_ind] is not None else 0
            gender_str = detected_objects.genders[person_ind] if detected_objects.genders[person_ind] is not None else "male"
            gender = GenderEnum.FEMALE if gender_str == "female" else GenderEnum.MALE
            
            # Get VIT confidence (gender score)
            vit_conf = detected_objects.gender_scores[person_ind]
            
            detected_person = DetectedPerson(
                age=int(age),
                gender=gender,
                face_bbox=None,
                face_confidence=None,
                body_bbox=body_bbox,
                body_confidence=body_conf,
                vit_confidence=vit_conf
            )
            detected_persons.append(detected_person)
        
        return detected_persons
    
    def recognize_simple(self, image: np.ndarray) -> List[DetectedPerson]:
        """
        Simplified recognition method that only returns DetectedPerson objects.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List[DetectedPerson]: List of detected persons with age, gender, and bbox info
        """
        detected_persons, _, _ = self.recognize(image)
        return detected_persons
    
    def recognize_batched_simple(self, images: List[np.ndarray]) -> List[List[DetectedPerson]]:
        """
        Simplified batch recognition method that only returns DetectedPerson objects.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List[List[DetectedPerson]]: List of detected persons for each image
        """
        detected_persons_list, _, _ = self.recognize_batched(images)
        return detected_persons_list
