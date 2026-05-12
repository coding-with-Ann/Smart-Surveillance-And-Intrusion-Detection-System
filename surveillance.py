"""Smart surveillance and intrusion detection using OpenCV.

The system detects motion with MOG2 background subtraction, cleans the mask,
tracks moving objects with simple centroid matching, and raises an alert when a
tracked object enters the restricted center zone.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]


@dataclass
class TrackedObject:
    """Store the current state of one tracked moving object."""

    object_id: int
    centroid: Point
    box: Box
    missed_frames: int = 0


class BackgroundSubtractor:
    """Create and clean a motion mask using MOG2 background subtraction."""

    def __init__(self) -> None:
        """Initialize the MOG2 subtractor and morphology kernel."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True,
        )
        self.kernel = np.ones((5, 5), dtype=np.uint8)

    def get_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a cleaned binary mask showing moving areas."""
        mask = self.subtractor.apply(frame)

        # Shadows are gray in MOG2 output. This keeps only stronger movement.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        return mask


class ObjectTracker:
    """Track moving objects between frames by matching centroid distances."""

    def __init__(self, distance_threshold: float = 50.0) -> None:
        """Create a tracker with a maximum matching distance."""
        self.distance_threshold = distance_threshold
        self.next_object_id = 1
        self.objects: Dict[int, TrackedObject] = {}

    def update(
        self,
        detections: List[Tuple[Point, Box]],
    ) -> Dict[int, TrackedObject]:
        """Match current detections to existing objects and return tracks."""
        unmatched_ids = set(self.objects.keys())

        for centroid, box in detections:
            matched_id = self._find_best_match(centroid, unmatched_ids)

            if matched_id is None:
                self._register_object(centroid, box)
                continue

            self.objects[matched_id].centroid = centroid
            self.objects[matched_id].box = box
            self.objects[matched_id].missed_frames = 0
            unmatched_ids.remove(matched_id)

        for object_id in unmatched_ids:
            self.objects[object_id].missed_frames += 1

        self._remove_lost_objects()
        return self.objects

    def _find_best_match(
        self,
        centroid: Point,
        candidate_ids: set[int],
    ) -> Optional[int]:
        """Return the closest object ID within the distance threshold."""
        best_id = None
        best_distance = self.distance_threshold

        for object_id in candidate_ids:
            old_centroid = self.objects[object_id].centroid
            distance = self._distance(centroid, old_centroid)

            if distance < best_distance:
                best_distance = distance
                best_id = object_id

        return best_id

    def _register_object(self, centroid: Point, box: Box) -> None:
        """Add a new tracked object."""
        self.objects[self.next_object_id] = TrackedObject(
            object_id=self.next_object_id,
            centroid=centroid,
            box=box,
        )
        self.next_object_id += 1

    def _remove_lost_objects(self) -> None:
        """Remove objects that have disappeared for several frames."""
        lost_ids = [
            object_id
            for object_id, tracked in self.objects.items()
            if tracked.missed_frames > 10
        ]

        for object_id in lost_ids:
            del self.objects[object_id]

    @staticmethod
    def _distance(point_a: Point, point_b: Point) -> float:
        """Return the Euclidean distance between two points."""
        return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))


class IntrusionDetector:
    """Check whether tracked objects enter the restricted zone."""

    def __init__(self) -> None:
        """Create an intrusion detector."""
        self.zone: Box = (0, 0, 0, 0)

    def update_zone(self, frame_width: int, frame_height: int) -> Box:
        """Set the restricted zone to the center 50 percent of the frame."""
        x1 = int(frame_width * 0.25)
        y1 = int(frame_height * 0.25)
        x2 = int(frame_width * 0.75)
        y2 = int(frame_height * 0.75)
        self.zone = (x1, y1, x2, y2)
        return self.zone

    def has_intrusion(self, objects: Dict[int, TrackedObject]) -> bool:
        """Return True if any object centroid is inside the restricted zone."""
        x1, y1, x2, y2 = self.zone

        for tracked in objects.values():
            center_x, center_y = tracked.centroid
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                return True

        return False


def find_moving_objects(mask: np.ndarray) -> List[Tuple[Point, Box]]:
    """Find moving object detections from the motion mask."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    detections: List[Tuple[Point, Box]] = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        centroid = (x + width // 2, y + height // 2)
        detections.append((centroid, (x, y, width, height)))

    return detections


def draw_scene(
    frame: np.ndarray,
    objects: Dict[int, TrackedObject],
    zone: Box,
    intrusion_detected: bool,
) -> None:
    """Draw tracks, the restricted zone, and the intrusion alert."""
    for tracked in objects.values():
        x, y, width, height = tracked.box
        center_x, center_y = tracked.centroid

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"ID {tracked.object_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if intrusion_detected:
        cv2.putText(
            frame,
            "INTRUSION DETECTED!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )


def run_surveillance() -> None:
    """Start the webcam loop for surveillance monitoring."""
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not access the webcam.")

    background = BackgroundSubtractor()
    tracker = ObjectTracker(distance_threshold=50.0)
    intrusion = IntrusionDetector()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            height, width = frame.shape[:2]
            mask = background.get_motion_mask(frame)
            detections = find_moving_objects(mask)
            tracked_objects = tracker.update(detections)
            zone = intrusion.update_zone(width, height)
            is_intrusion = intrusion.has_intrusion(tracked_objects)

            draw_scene(frame, tracked_objects, zone, is_intrusion)

            cv2.imshow("Smart Surveillance", frame)
            cv2.imshow("Motion Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_surveillance()
