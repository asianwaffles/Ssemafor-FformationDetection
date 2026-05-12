from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

try:
    from .pose_extractor import PoseLocalizer3DNode
except ImportError:
    PoseLocalizer3DNode = None

class FormationType(Enum):
    VIS_A_VIS  = "vis-a-vis"
    L_SHAPED   = "L-shaped"
    SIDE_BY_SIDE = "side-by-side"
    CIRCULAR   = "circular"
    NONE       = "none"


@dataclass
class FormationResult:
    """Full output of one detection pass."""
    formation_type: FormationType

    # Openness
    openness_score: float        # 0–1  (1 = maximally open/accessible)
    angular_gap_deg: float       # largest unblocked arc around o-space (degrees)
    min_reorientation_cost: float # sum of rotation each person needs for best approach (degrees)
    best_approach_angle: float   # compass angle robot should approach from (degrees, 0=east)

    # Raw geometry (useful for downstream predictor)
    center: np.ndarray           # o-space centroid (2-D floor-plane coords)
    inter_person_distances: np.ndarray  # NxN distance matrix


# ─────────────────────────────────────────────
#  Detector
# ─────────────────────────────────────────────

class FFormationDetector:
    """
    Detects F-formations in 2-D (bounding boxes) or 3-D (MediaPipe poses)
    and computes a composite openness score for robot intervention timing.

    Openness score combines:
      • Angular gap  – largest unobstructed arc around the shared o-space
      • Reorientation cost – minimum total rotation needed for the group
                             to naturally include a new participant

    Both metrics are normalised to [0, 1] and combined via configurable weights.
    """

    # Shoulder half-width used for arc-blocking calculation
    SHOULDER_HALF_WIDTH_3D = 0.225   # metres  (~0.45 m total)
    SHOULDER_HALF_WIDTH_2D = 25.0    # pixels  (~50 px total)

    # Angular gap normalisation cap: a gap of this size → gap_score = 1.0
    GAP_NORM_DEG = 120.0

    # Number of candidate approach angles sampled around the group
    N_APPROACH_CANDIDATES = 36

    def __init__(
        self,
        use_3d: bool = False,
        distance_threshold: Optional[float] = None,
        angle_threshold: float = 60.0,
        openness_weight_gap: float = 0.55,
        openness_weight_cost: float = 0.45,
    ):
        """
        Args:
            use_3d: Use MediaPipe 3-D poses instead of 2-D bounding boxes.
            distance_threshold: Max distance between people to be considered
                a formation. Defaults to 1.5 m (3-D) or 300 px (2-D).
            angle_threshold: Tolerance (degrees) for facing-direction checks.
            openness_weight_gap: Weight for angular-gap component (0–1).
            openness_weight_cost: Weight for reorientation-cost component (0–1).
                The two weights should sum to 1.
        """
        self.use_3d = use_3d
        self.distance_threshold = distance_threshold or (1.5 if use_3d else 300.0)
        self.angle_threshold = angle_threshold
        self.w_gap  = openness_weight_gap
        self.w_cost = openness_weight_cost

        self.pose_extractor = None
        if use_3d and PoseLocalizer3DNode:
            self.pose_extractor = PoseLocalizer3DNode()

    def detect_formation(
        self,
        image: np.ndarray,
        bboxes: list,
        positions: np.ndarray,
        landmarks: Optional[list] = None,
    ) -> FormationResult:
        """
        Main entry point.  Automatically selects 2-D or 3-D path.

        Args:
            image:     BGR image (used only in 2-D mode).
            bboxes:    Bounding boxes [x1,y1,x2,y2,...] – 2-D mode.
            positions: (N,2) pixel positions OR (N,3) metric positions.
            landmarks: MediaPipe landmark arrays – 3-D mode.

        Returns:
            FormationResult with formation type, openness metrics, and geometry.
        """
        is_3d = (
            positions.ndim == 2
            and positions.shape[1] == 3
            and not np.allclose(positions[:, 2], 0)
        )

        if is_3d and landmarks is not None and self.pose_extractor:
            return self._detect_3d(positions, landmarks)
        else:
            return self._detect_2d(image, bboxes, positions)
            
    def _detect_3d(self, positions_3d: np.ndarray, landmarks: list) -> FormationResult:
        body_yaws = np.array([
            self.pose_extractor.get_body_orientation_3d(lm) for lm in landmarks
        ])
        distances = np.linalg.norm(positions_3d[:, None] - positions_3d, axis=2)

        formation_type = FormationType.NONE

        if len(positions_3d) == 2:
            formation_type = self._classify_pair_3d(positions_3d, body_yaws, distances)

        elif len(positions_3d) >= 3:
            formation_type = self._classify_group_3d(positions_3d, body_yaws, distances)

        # Openness uses floor-plane (x, y) only — z is irrelevant for accessibility
        floor_positions = positions_3d[:, :2]
        openness = self._compute_openness(floor_positions, body_yaws)

        result = FormationResult(
            formation_type=formation_type,
            openness_score=openness["score"],
            angular_gap_deg=openness["angular_gap"],
            min_reorientation_cost=openness["min_cost"],
            best_approach_angle=openness["best_angle"],
            center=openness["center"],
            inter_person_distances=distances,
        )
        self._log(result)
        return result

    def _classify_pair_3d(self, positions, yaws, distances) -> FormationType:
        d = distances[0, 1]
        if d >= self.distance_threshold:
            return FormationType.NONE

        p1_to_p2 = positions[1] - positions[0]
        interaction_angle = np.degrees(np.arctan2(p1_to_p2[0], p1_to_p2[2]))

        p1_facing = self._is_facing(yaws[0], interaction_angle)
        p2_facing = self._is_facing(yaws[1], interaction_angle + 180)

        yaw_diff = self._angle_diff(yaws[0], yaws[1])

        if p1_facing and p2_facing:
            return FormationType.VIS_A_VIS
        elif yaw_diff < 45:
            return FormationType.SIDE_BY_SIDE
        elif 45 <= yaw_diff <= 135:
            return FormationType.L_SHAPED
        else:
            return FormationType.SIDE_BY_SIDE

    def _classify_group_3d(self, positions, yaws, distances) -> FormationType:
        max_distance = np.max(distances[distances > 0])
        if max_distance >= self.distance_threshold:
            return FormationType.NONE

        center = positions.mean(axis=0)
        dists_to_center = np.linalg.norm(positions - center, axis=1)
        std_dist  = np.std(dists_to_center)
        mean_dist = np.mean(dists_to_center)

        # Circular: equidistant from center + mostly facing inward
        if std_dist < mean_dist * 0.4:
            facing_count = sum(
                self._is_facing(
                    yaws[i],
                    np.degrees(np.arctan2(
                        (center - positions[i])[0],
                        (center - positions[i])[2]
                    ))
                )
                for i in range(len(positions))
            )
            if facing_count >= len(positions) * 0.6:
                return FormationType.CIRCULAR

        return FormationType.SIDE_BY_SIDE
        
    def _detect_2d(self, image, bboxes, positions) -> FormationResult:
        distances = np.linalg.norm(positions[:, None] - positions, axis=2)
        orientations = [self._estimate_orientation_2d(bbox, image) for bbox in bboxes]

        # For 2-D we don't have true yaw; use relative position angle as a proxy
        # so openness math still works (yaw = angle from centroid outward)
        center_2d = positions.mean(axis=0)
        proxy_yaws = np.array([
            np.degrees(np.arctan2(
                (pos - center_2d)[1],
                (pos - center_2d)[0]
            )) % 360
            for pos in positions
        ])

        formation_type = FormationType.NONE

        if len(bboxes) == 2:
            formation_type = self._classify_pair_2d(positions, distances)
        elif len(bboxes) >= 3:
            formation_type = self._classify_group_2d(positions, distances)

        openness = self._compute_openness(positions, proxy_yaws, use_3d=False)

        result = FormationResult(
            formation_type=formation_type,
            openness_score=openness["score"],
            angular_gap_deg=openness["angular_gap"],
            min_reorientation_cost=openness["min_cost"],
            best_approach_angle=openness["best_angle"],
            center=openness["center"],
            inter_person_distances=distances,
        )
        self._log(result)
        return result

    def _classify_pair_2d(self, positions, distances) -> FormationType:
        d = distances[0, 1]
        if d >= self.distance_threshold:
            return FormationType.NONE

        delta = positions[1] - positions[0]
        angle = np.degrees(np.arctan2(delta[1], delta[0]))

        if abs(angle) < 30 or abs(abs(angle) - 180) < 30:
            return FormationType.SIDE_BY_SIDE
        elif abs(abs(angle) - 90) < 30:
            return FormationType.VIS_A_VIS
        else:
            return FormationType.L_SHAPED

    def _classify_group_2d(self, positions, distances) -> FormationType:
        if np.max(distances[distances > 0]) >= self.distance_threshold:
            return FormationType.NONE

        center = positions.mean(axis=0)
        dists_to_center = np.linalg.norm(positions - center, axis=1)
        if np.std(dists_to_center) < np.mean(dists_to_center) * 0.5:
            return FormationType.CIRCULAR
        return FormationType.SIDE_BY_SIDE
        
    def _compute_openness(
        self,
        positions: np.ndarray,   # (N, 2) floor-plane
        body_yaws: np.ndarray,   # (N,)  degrees
        use_3d: Optional[bool] = None,
    ) -> dict:
        """
        Compute composite openness score.

        Returns dict with keys:
            score, angular_gap, min_cost, best_angle, center
        """
        if use_3d is None:
            use_3d = self.use_3d

        center = positions.mean(axis=0)
        angular_gap = self._angular_gap(positions, center, use_3d)
        min_cost, best_angle = self._reorientation_cost(positions, body_yaws, center)

        # Normalise both to [0, 1]
        gap_score  = min(angular_gap / self.GAP_NORM_DEG, 1.0)
        max_cost   = 180.0 * len(positions)
        cost_score = 1.0 - min(min_cost / max_cost, 1.0)

        score = self.w_gap * gap_score + self.w_cost * cost_score

        return {
            "score":       float(score),
            "angular_gap": float(angular_gap),
            "min_cost":    float(min_cost),
            "best_angle":  float(best_angle),
            "center":      center,
        }

    def _angular_gap(
        self,
        positions: np.ndarray,
        center: np.ndarray,
        use_3d: bool,
    ) -> float:
        """
        Largest unblocked arc (degrees) around the o-space perimeter.

        Each person blocks an arc proportional to shoulder width / distance
        from center.  Returns 360 if fewer than 2 people are present.
        """
        shoulder_hw = self.SHOULDER_HALF_WIDTH_3D if use_3d else self.SHOULDER_HALF_WIDTH_2D

        blocked = []
        for pos in positions:
            to_person = pos - center
            dist = np.linalg.norm(to_person)
            if dist < 1e-6:
                continue
            angle = np.degrees(np.arctan2(to_person[1], to_person[0])) % 360
            half_block = np.degrees(np.arctan2(shoulder_hw, dist))
            blocked.append((angle, half_block))

        if len(blocked) < 2:
            return 360.0

        blocked.sort(key=lambda x: x[0])

        largest_gap = 0.0
        n = len(blocked)
        for i in range(n):
            curr_end   = (blocked[i][0]           + blocked[i][1])           % 360
            next_start = (blocked[(i + 1) % n][0] - blocked[(i + 1) % n][1]) % 360
            gap = (next_start - curr_end) % 360
            largest_gap = max(largest_gap, gap)

        return largest_gap

    def _reorientation_cost(
        self,
        positions: np.ndarray,
        body_yaws: np.ndarray,
        center: np.ndarray,
    ) -> tuple[float, float]:
        """
        Sample N_APPROACH_CANDIDATES positions around the group and find the
        one minimising total group reorientation cost.

        Returns:
            (min_total_cost_degrees, best_approach_angle_degrees)
        """
        radii = np.linalg.norm(positions - center, axis=1)
        approach_radius = radii.max() * 1.4

        min_cost   = float("inf")
        best_angle = 0.0

        for i in range(self.N_APPROACH_CANDIDATES):
            approach_deg = (360.0 / self.N_APPROACH_CANDIDATES) * i
            approach_rad = np.radians(approach_deg)
            robot_pos = center + approach_radius * np.array([
                np.cos(approach_rad),
                np.sin(approach_rad),
            ])

            total_cost = 0.0
            for j, pos in enumerate(positions):
                to_robot = robot_pos - pos
                desired_yaw = np.degrees(np.arctan2(to_robot[1], to_robot[0])) % 360
                cost = self._angle_diff(body_yaws[j], desired_yaw)
                total_cost += cost
            if total_cost < min_cost:
                min_cost   = total_cost
                best_angle = approach_deg
        return min_cost, best_angle

    def _is_facing(self, body_yaw: float, target_yaw: float) -> bool:
        return self._angle_diff(body_yaw, target_yaw) < self.angle_threshold

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Smallest absolute difference between two angles (degrees)."""
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    def _estimate_orientation_2d(self, bbox, image: np.ndarray) -> float:
        """
        Rough front/back estimate from pixel variance in the upper third
        of the bounding box.  Returns 0 (back) – 1 (front).
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        upper = gray[:max(1, gray.shape[0] // 3), :]
        return float(min(np.var(upper) / 1000.0, 1.0))

    @staticmethod
    def _log(result: FormationResult) -> None:
        print(
            f"  Formation : {result.formation_type.value}\n"
            f"  Openness  : {result.openness_score:.2f}  "
            f"(gap {result.angular_gap_deg:.1f}°, "
            f"cost {result.min_reorientation_cost:.1f}°)\n"
            f"  Best approach angle: {result.best_approach_angle:.1f}°"
        )
