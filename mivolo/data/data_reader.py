from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

IMAGES_EXT: Tuple = (".jpeg", ".jpg", ".png", ".webp", ".bmp", ".gif")
VIDEO_EXT: Tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class PictureInfo:
    image_path: str
    age: Optional[str]  # age or age range(start;end format) or "-1"
    gender: Optional[str]  # "M" of "F" or "-1"
    bbox: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])  # face bbox: xyxy
    person_bbox: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])  # person bbox: xyxy

    @property
    def has_person_bbox(self) -> bool:
        return any(coord != -1 for coord in self.person_bbox)

    @property
    def has_face_bbox(self) -> bool:
        return any(coord != -1 for coord in self.bbox)

    def has_gt(self, only_age: bool = False) -> bool:
        if only_age:
            return self.age != "-1"
        else:
            return not (self.age == "-1" and self.gender == "-1")

    def clear_person_bbox(self):
        self.person_bbox = [-1, -1, -1, -1]

    def clear_face_bbox(self):
        self.bbox = [-1, -1, -1, -1]


class AnnotType(Enum):
    ORIGINAL = "original"
    PERSONS = "persons"
    NONE = "none"

    @classmethod
    def _missing_(cls, value):
        print(f"WARN: Unknown annotation type {value}.")
        return AnnotType.NONE


def get_all_files(path: str, extensions: Tuple = IMAGES_EXT) -> List[str]:
    """Recursively collect files with specified extensions."""
    path_obj = Path(path)
    extensions = tuple(ext.lower() for ext in extensions)
    return [
        str(p)
        for p in path_obj.rglob("*")
        if p.is_file() and "directory" not in p.name and p.suffix.lower() in extensions
    ]


class InputType(Enum):
    Image = 0
    Video = 1
    VideoStream = 2


def get_input_type(input_path: str) -> InputType:
    """Determine the type of the provided input path or URL."""
    path_obj = Path(input_path)
    if path_obj.is_dir():
        print("Input is a folder, only images will be processed")
        return InputType.Image
    if path_obj.is_file():
        suffix = path_obj.suffix.lower()
        if suffix in VIDEO_EXT:
            return InputType.Video
        if suffix in IMAGES_EXT:
            return InputType.Image
        raise ValueError(
            f"Unknown or unsupported input file format {input_path}, supported video formats: {VIDEO_EXT}, supported image formats: {IMAGES_EXT}"
        )
    if input_path.startswith("http") and not any(input_path.endswith(ext) for ext in IMAGES_EXT):
        return InputType.VideoStream
    raise ValueError(f"Unknown input {input_path}")


def read_csv_annotation_file(
    annotation_file: str, images_dir: str, ignore_without_gt: bool = False
) -> Tuple[Dict[str, List[PictureInfo]], AnnotType]:
    """Read annotation CSV and collect picture information."""
    bboxes_per_image: Dict[str, List[PictureInfo]] = defaultdict(list)

    annotation_path = Path(annotation_file)
    images_path = Path(images_dir)
    df = pd.read_csv(annotation_path, sep=",")

    annot_type = AnnotType("persons") if "person_x0" in df.columns else AnnotType("original")
    print(f"Reading {annotation_path} (type: {annot_type})...")

    missing_images = 0
    for _, row in df.iterrows():
        img_path = images_path / row["img_name"]
        if not img_path.exists():
            missing_images += 1
            continue

        face_x1, face_y1, face_x2, face_y2 = row["face_x0"], row["face_y0"], row["face_x1"], row["face_y1"]
        age, gender = str(row["age"]), str(row["gender"])

        if ignore_without_gt and (age == "-1" or gender == "-1"):
            continue

        if annot_type == AnnotType.PERSONS:
            p_x1, p_y1, p_x2, p_y2 = row["person_x0"], row["person_y0"], row["person_x1"], row["person_y1"]
            person_bbox = list(map(int, [p_x1, p_y1, p_x2, p_y2]))
        else:
            person_bbox = [-1, -1, -1, -1]

        bbox = list(map(int, [face_x1, face_y1, face_x2, face_y2]))
        pic_info = PictureInfo(str(img_path), age, gender, bbox, person_bbox)
        assert isinstance(pic_info.person_bbox, list)

        bboxes_per_image[str(img_path)].append(pic_info)

    if missing_images > 0:
        print(f"WARNING: Missing images: {missing_images}/{len(df)}")
    return bboxes_per_image, annot_type
