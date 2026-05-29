import cv2
import numpy as np
import os
import json
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

@dataclass
class RingMetrics:
    is_closed: bool
    circularity_score: float
    radius_px: float
    centroid: Tuple[int, int]

@dataclass
class NormalizedToken:
    token_id: int
    centroid: Tuple[int, int]
    placement_angle_rad: float
    matrix_outward: np.ndarray  # 28x28 float32 array [0.0 - 1.0]
    matrix_inward: np.ndarray   # 28x28 float32 array [0.0 - 1.0]

@dataclass
class ParsedSymbol:
    symbol_class: str           # e.g., 'fire_sigil', 'arrow_sign', 'unknown'
    confidence_score: float     # Bound float [0.0 - 1.0]
    placement_angle_rad: float  # Angle relative to ring center
    orientation: str            # Strictly literal: 'outward' or 'inward'

@dataclass
class FinalSpellIR:
    is_valid_spell: bool
    ring_circularity: float
    detected_symbols: List[ParsedSymbol]

class DeterministicSpellParser:
    """
    Deterministic Contour-Driven Spell Recognition Engine.
    Translates freehand canvas stroke inputs into deterministic gameplay parameters (Spell IR).
    Adheres strictly to continuous vector mathematics and topological invariant properties.
    """
    def __init__(self, circularity_threshold: float = 0.72):
        self.circularity_threshold = circularity_threshold
        self.preloaded_templates = {}

    def zhang_suen_thinning(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Applies high-performance, fully-vectorized Zhang-Suen thinning algorithm to reduce
        foreground contours (255) to 1-pixel wide centerlines.
        
        Mathematical Rationale:
        Eliminates stroke-width bias by reducing paths to their topological skeletons.
        A pixel p1 is cleared in sub-iteration 1 if:
          - 2 <= B(p1) <= 6
          - A(p1) == 1
          - p2 * p4 * p6 == 0
          - p4 * p6 * p8 == 0
        And in sub-iteration 2, the last two conditions become:
          - p2 * p4 * p8 == 0
          - p2 * p6 * p8 == 0
        Where B(p1) is the count of non-zero 8-neighbors of p1, and A(p1) is the count of
        0-to-1 transitions in the ordered neighbors list: p2, p3, p4, p5, p6, p7, p8, p9, p2.
        """
        # Normalize to binary values 0 and 1
        im = (binary_mask > 0).astype(np.uint8)
        
        while True:
            # Pad image to safely compute shifts for boundaries
            padded = np.pad(im, 1, mode='constant', constant_values=0)
            
            # Extract 8-connected neighbors relative to target p1
            p2 = padded[0:-2, 1:-1]
            p3 = padded[0:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, 0:-2]
            p8 = padded[1:-1, 0:-2]
            p9 = padded[0:-2, 0:-2]
            p1 = padded[1:-1, 1:-1]
            
            # Count of non-zero neighbors (B value)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            
            # Count of 0-to-1 transitions in ordered circle (A value)
            a = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + \
                ((p3 == 0) & (p4 == 1)).astype(np.uint8) + \
                ((p4 == 0) & (p5 == 1)).astype(np.uint8) + \
                ((p5 == 0) & (p6 == 1)).astype(np.uint8) + \
                ((p6 == 0) & (p7 == 1)).astype(np.uint8) + \
                ((p7 == 0) & (p8 == 1)).astype(np.uint8) + \
                ((p8 == 0) & (p9 == 1)).astype(np.uint8) + \
                ((p9 == 0) & (p2 == 1)).astype(np.uint8)
                
            # Conditions for sub-iteration 1 deletion
            cond1 = (p1 == 1)
            cond2 = (b >= 2) & (b <= 6)
            cond3 = (a == 1)
            cond4 = (p2 * p4 * p6 == 0)
            cond5 = (p4 * p6 * p8 == 0)
            
            del_mask = cond1 & cond2 & cond3 & cond4 & cond5
            im_step1 = im.copy()
            im_step1[del_mask] = 0
            
            # Re-pad and evaluate sub-iteration 2 on updated image
            padded = np.pad(im_step1, 1, mode='constant', constant_values=0)
            p2 = padded[0:-2, 1:-1]
            p3 = padded[0:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, 0:-2]
            p8 = padded[1:-1, 0:-2]
            p9 = padded[0:-2, 0:-2]
            p1 = padded[1:-1, 1:-1]
            
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            
            a = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + \
                ((p3 == 0) & (p4 == 1)).astype(np.uint8) + \
                ((p4 == 0) & (p5 == 1)).astype(np.uint8) + \
                ((p5 == 0) & (p6 == 1)).astype(np.uint8) + \
                ((p6 == 0) & (p7 == 1)).astype(np.uint8) + \
                ((p7 == 0) & (p8 == 1)).astype(np.uint8) + \
                ((p8 == 0) & (p9 == 1)).astype(np.uint8) + \
                ((p9 == 0) & (p2 == 1)).astype(np.uint8)
                
            cond1 = (p1 == 1)
            cond2 = (b >= 2) & (b <= 6)
            cond3 = (a == 1)
            cond4 = (p2 * p4 * p8 == 0)
            cond5 = (p2 * p6 * p8 == 0)
            
            del_mask2 = cond1 & cond2 & cond3 & cond4 & cond5
            im_step2 = im_step1.copy()
            im_step2[del_mask2] = 0
            
            # Terminate when no more pixels are deleted (steady-state skeleton achieved)
            if np.array_equal(im, im_step2):
                break
                
            im = im_step2
            
        return (im * 255).astype(np.uint8)

    def calculate_median_stroke_width(self, binary_mask: np.ndarray) -> float:
        """
        Calculates the dynamic median stroke width of the drawing using Distance Transform.
        
        Mathematical Rationale:
        Applies L2 Distance Transform on the foreground binary mask. For every foreground pixel,
        the L2 value represents the shortest Euclidean distance to the background boundary.
        By sampling these values exclusively along the 1-pixel skeleton centerline, we capture the
        exact local stroke radius (half of the full stroke width).
        Multiplying by 2.0 and computing the median yields a global stroke width metric invariant
        to localized hand tremors, tapering, and intersections.
        """
        if np.count_nonzero(binary_mask) == 0:
            return 1.0
            
        # 1. Distance transform (L2 metric with 5x5 mask for higher precision)
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # 2. Extract 1-pixel skeleton centerline
        skeleton = self.zhang_suen_thinning(binary_mask)
        
        # 3. Sample values at skeleton locations
        centerline_distances = dist_transform[skeleton > 0]
        
        if len(centerline_distances) == 0:
            return 1.0
            
        # Stroke width is twice the distance from centerline to closest background boundary
        stroke_widths = centerline_distances * 2.0
        
        # 4. Compute median to suppress outlier junctions or noise
        median_width = float(np.median(stroke_widths))
        
        return max(median_width, 1.0)

    def bridge_gaps(self, binary_mask: np.ndarray, skeleton: np.ndarray, median_width: float) -> np.ndarray:
        """
        Applies the Gap-Bridging Engine to close minor stroke interruptions.
        
        Mathematical Rationale:
        1. Detects endpoints on the 1-pixel skeleton using a 255-scale 3x3 convolution filter:
           A pixel is an endpoint if skeleton[y, x] == 255 AND sum(8-neighbors) == 255.
        2. Calculates pairwise Euclidean distances between all detected endpoints.
        3. Bridges endpoint pairs where distance <= tau_gap = 4.5 * median_width.
        4. Connections are drawn with `cv2.line` directly onto the binary mask, with
           thickness matching the median stroke width, sealing any boundary leaks.
        """
        # Define 3x3 convolution kernel to sum 8 neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        
        # Convolve skeleton (convert to float32 to prevent uint8 overflow saturation)
        skel_float32 = skeleton.astype(np.float32)
        neighbor_sum = cv2.filter2D(skel_float32, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        
        # Identify coordinates meeting 255-scale constraints (exactly one neighbor of value 255)
        endpoints_indices = np.argwhere((skeleton == 255) & (np.isclose(neighbor_sum, 255.0)))
        
        if len(endpoints_indices) < 2:
            return binary_mask.copy()
            
        # Convert coordinate arrays to list of tuples (x, y)
        endpoints = [(int(pt[1]), int(pt[0])) for pt in endpoints_indices]
        
        n = len(endpoints)
        bridged_mask = binary_mask.copy()
        tau_gap = median_width * 4.5
        
        connected = set()
        
        # Pairwise matching of endpoints within threshold
        for i in range(n):
            if i in connected:
                continue
            p_i = endpoints[i]
            
            best_j = -1
            best_dist = float('inf')
            
            for j in range(n):
                if i == j or j in connected:
                    continue
                p_j = endpoints[j]
                dist = float(np.sqrt((p_i[0] - p_j[0])**2 + (p_i[1] - p_j[1])**2))
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                    
            if best_j != -1 and best_dist <= tau_gap:
                # Bridge gap directly in the binary mask using cv2.line
                thickness = int(max(round(median_width), 1))
                cv2.line(bridged_mask, p_i, endpoints[best_j], 255, thickness)
                connected.add(i)
                connected.add(best_j)
                
        return bridged_mask

    def verify_circular_leakage(self, contour: np.ndarray, median_width: float, shape: Tuple[int, int]) -> Tuple[bool, float]:
        """
        Verifies if the candidate ring has a boundary leak (is open) by analyzing the isolated skeleton.
        
        Mathematical Rationale:
        1. Draw the contour on a blank mask with a stroke thickness matching the median stroke width.
        2. Skeletonize this isolated mask.
        3. Locate endpoints on the isolated skeleton using the 255-scale endpoint detector.
        4. If there are 0 endpoints, it is a perfect topological closed ring (genus-1 loop).
        5. If there are exactly 2 endpoints, calculate the Euclidean distance between them.
           If the distance > median_width * 2.0, the ring has an unclosed boundary leak.
        """
        # 1. Isolate the candidate shape contour as a standalone binary mask
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=int(max(round(median_width), 1)))
        
        # 2. Skeletonize isolated mask to locate ends
        skeleton = self.zhang_suen_thinning(mask)
        
        # 3. Locate endpoints
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        skel_float32 = skeleton.astype(np.float32)
        neighbor_sum = cv2.filter2D(skel_float32, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.argwhere((skeleton == 255) & (np.isclose(neighbor_sum, 255.0)))
        
        # 4. Evaluate closure & calculate gap
        if len(endpoints) == 0:
            return True, 0.0
        elif len(endpoints) == 2:
            pt1 = endpoints[0]
            pt2 = endpoints[1]
            dist = float(np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))
            is_closed = dist <= (median_width * 2.0)
            return is_closed, dist
        else:
            # For complex skeletons, compute the minimum distance between any pair of endpoints
            if len(endpoints) < 2:
                return True, 0.0
            min_dist = float('inf')
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    d = np.sqrt((endpoints[i][0] - endpoints[j][0])**2 + (endpoints[i][1] - endpoints[j][1])**2)
                    if d < min_dist:
                        min_dist = d
            is_closed = min_dist <= (median_width * 2.0)
            return is_closed, float(min_dist)

    def isolate_parent_ring(self, image_array: np.ndarray, debug_inspect: bool = False) -> Tuple[Optional[np.ndarray], Optional[RingMetrics]]:
        """
        Executes Step 1 & Step 2: Binarization, Skeletonization, Contour Extraction, 
        Circularity Verification, and Convex Hull Gap Analysis.
        Returns the isolated parent ring contour array and its corresponding RingMetrics.
        """
        if image_array is None or image_array.size == 0:
            return None, None
            
        # --- Step 1: Preprocessing & Skeletonization ---
        # 1. Grayscale Conversion
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array.copy()
            
        # 2. Otsu's Adaptive Thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Adaptive Inversion: ensure strokes are 255 (white) and background is 0 (black)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
            
        # 3. Dynamic Median Stroke Width Calculation
        median_width = self.calculate_median_stroke_width(binary)
        
        # 4. Generate Skeleton and Apply Gap-Bridging Engine
        skeleton = self.zhang_suen_thinning(binary)
        bridged_binary = self.bridge_gaps(binary, skeleton, median_width)
        
        # --- Step 2: Parent Ring Isolation ---
        # Extract contours using Hierarchical Tracking (RETR_TREE, CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(bridged_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours or hierarchy is None:
            return None, None
            
        valid_candidates = []
        
        for idx, contour in enumerate(contours):
            # Calculate Area (A) and Perimeter (P)
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, closed=True))
            
            if perimeter == 0:
                continue
                
            # Compute Circularity Index via Isoperimetric Inequality: S = (4 * pi * Area) / (Perimeter^2)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # Validate against circularity index threshold
            if circularity >= self.circularity_threshold:
                # Perform Circular Leakage Verification on the isolated shape
                is_closed, gap_dist = self.verify_circular_leakage(contour, median_width, gray.shape)
                
                # Fetch Centroid (Cx, Cy) using spatial moments
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = np.mean(contour, axis=0)[0].astype(int)
                    
                # Fetch Physical Radius using standard min enclosing circle bounds
                _, radius = cv2.minEnclosingCircle(contour)
                
                valid_candidates.append({
                    'contour': contour,
                    'area': area,
                    'metrics': RingMetrics(
                        is_closed=is_closed,
                        circularity_score=circularity,
                        radius_px=float(radius),
                        centroid=(cx, cy)
                    )
                })
                
        if not valid_candidates:
            return None, None
            
        # Double-Ring/Concentric Loop Resolution:
        # Sort candidates descending by area. The largest area contour is the Topological Root.
        valid_candidates.sort(key=lambda x: x['area'], reverse=True)
        parent_ring = valid_candidates[0]
        
        parent_contour = parent_ring['contour']
        parent_metrics = parent_ring['metrics']
        
        # --- Debug Inspection Layer ---
        if debug_inspect:
            os.makedirs('./debug_output', exist_ok=True)
            
            # Grayscale for visual overlays
            grayscale_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Visual 1: Skeleton and endpoints highlight
            v1 = grayscale_bgr.copy()
            # Overlay skeleton in bright green
            v1[skeleton > 0] = [0, 255, 0]
            # Locate and draw endpoints in bright red
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.uint8)
            skel_float32 = skeleton.astype(np.float32)
            neighbor_sum = cv2.filter2D(skel_float32, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            endpoints_indices = np.argwhere((skeleton == 255) & (np.isclose(neighbor_sum, 255.0)))
            for pt in endpoints_indices:
                cv2.circle(v1, (int(pt[1]), int(pt[0])), radius=4, color=(0, 0, 255), thickness=-1)
            cv2.imwrite('./debug_output/01_skeleton_endpoints.png', v1)
            
            # Visual 2: Isolated parent ring with bounding box, centroid, and circularity score
            v2 = grayscale_bgr.copy()
            # Draw parent contour in blue
            cv2.drawContours(v2, [parent_contour], -1, (255, 0, 0), thickness=2)
            # Draw centroid in green
            cx, cy = parent_metrics.centroid
            cv2.circle(v2, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(parent_contour)
            cv2.rectangle(v2, (x, y), (x + w, y + h), (0, 255, 255), thickness=1)
            # Annotate metrics text
            label_text = f"S: {parent_metrics.circularity_score:.3f} | C: ({cx}, {cy})"
            cv2.putText(v2, label_text, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite('./debug_output/02_isolated_parent_ring.png', v2)
            
            # Visual 3: Concentric elements and spatial vectors
            v3 = grayscale_bgr.copy()
            cv2.drawContours(v3, [parent_contour], -1, (255, 0, 0), thickness=2)
            cv2.circle(v3, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)
            # If multiple concentric rings are present, draw them in purple and draw an arrow
            if len(valid_candidates) > 1:
                for child in valid_candidates[1:]:
                    child_contour = child['contour']
                    cv2.drawContours(v3, [child_contour], -1, (255, 0, 255), thickness=1)
                    ccx, ccy = child['metrics'].centroid
                    cv2.circle(v3, (ccx, ccy), radius=3, color=(255, 0, 255), thickness=-1)
            cv2.imwrite('./debug_output/03_modifier_rois.png', v3)
            
        return parent_contour, parent_metrics

    def extract_and_normalize_tokens(
        self, 
        image_array: np.ndarray, 
        parent_ring_contour: np.ndarray, 
        ring_metrics: RingMetrics,
        debug_inspect: bool = False
    ) -> List[NormalizedToken]:
        """
        Executes Step 3 and Step 4 pipelines: Filters internal contours, clusters multi-stroke 
        fragments, calculates spatial centroids, applies affine de-rotation matrices, 
        and vectorizes tokens into standardized 28x28 float32 pairs.
        """
        if image_array is None or image_array.size == 0 or parent_ring_contour is None:
            return []

        # --- Grayscale and Binarization ---
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # Dynamic values based on stroke width
        median_width = self.calculate_median_stroke_width(binary)
        parent_area = float(cv2.contourArea(parent_ring_contour))
        
        # Calculate skeleton mask for centerline token cropping
        skeleton = self.zhang_suen_thinning(binary)

        # Extract all contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # --- Step 3: Leaf Symbol Spatial Enclosure Filtering ---
        child_candidates = []
        for idx, c in enumerate(contours):
            area = float(cv2.contourArea(c))
            # Exclude parent contour and very large concentric boundaries
            if area >= 0.8 * parent_area:
                continue

            # Calculate spatial moments centroid of candidate
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            # Adaptive boundary tolerance condition: pointPolygonTest >= -0.5 * W_m
            dist = cv2.pointPolygonTest(parent_ring_contour, (cx, cy), True)
            if dist >= -(median_width * 0.5):
                child_candidates.append((c, area, (cx, cy)))

        if not child_candidates:
            return []

        # --- Step 3: Hierarchical Agglomerative Clustering (HAC) for Multi-Stroke Grouping ---
        def bbox_clearance_dist(box_a, box_b):
            x1_a, y1_a, w_a, h_a = box_a
            x1_b, y1_b, w_b, h_b = box_b
            
            x2_a = x1_a + w_a
            y2_a = y1_a + h_a
            x2_b = x1_b + w_b
            y2_b = y1_b + h_b
            
            dx = 0.0
            if x2_a < x1_b:
                dx = float(x1_b - x2_a)
            elif x2_b < x1_a:
                dx = float(x1_a - x2_b)
                
            dy = 0.0
            if y2_a < y1_b:
                dy = float(y1_b - y2_a)
            elif y2_b < y1_a:
                dy = float(y1_a - y2_b)
                
            return np.sqrt(dx**2 + dy**2)

        num_candidates = len(child_candidates)
        adj = {i: set() for i in range(num_candidates)}
        tau_group = 1.5 * median_width

        # Build proximity graph
        for i in range(num_candidates):
            box_i = cv2.boundingRect(child_candidates[i][0])
            for j in range(i + 1, num_candidates):
                box_j = cv2.boundingRect(child_candidates[j][0])
                if bbox_clearance_dist(box_i, box_j) <= tau_group:
                    adj[i].add(j)
                    adj[j].add(i)

        # Find connected components
        visited = set()
        groups = []
        for i in range(num_candidates):
            if i not in visited:
                group = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    group.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                groups.append(group)

        tokens = []
        token_bboxes = [] # Keep track for telemetry overlays

        # --- Step 4: Normalization (Scale & Rotation Invariance) ---
        for idx_token, group in enumerate(groups):
            group_contours = [child_candidates[idx][0] for idx in group]
            
            # Group / Concatenate contours to calculate definitive centroid
            unified_contour = np.concatenate(group_contours, axis=0)
            M = cv2.moments(unified_contour)
            if M["m00"] != 0:
                sx = float(M["m10"] / M["m00"])
                sy = float(M["m01"] / M["m00"])
            else:
                pts = unified_contour.reshape(-1, 2)
                sx, sy = np.mean(pts, axis=0)

            # Get tight bounding box of unified contours
            x_new, y_new, w_new, h_new = cv2.boundingRect(unified_contour)
            token_bboxes.append((x_new, y_new, w_new, h_new, sx, sy))

            # Displacement vector and radial angle calculation: theta = atan2(Sy - Cy, Sx - Cx)
            cx, cy = ring_metrics.centroid
            theta = float(np.arctan2(sy - cy, sx - cx))

            # Localized ROI Padding to prevent warping clipping
            pad = int(round(2.0 * median_width))
            x_min = max(0, x_new - pad)
            y_min = max(0, y_new - pad)
            x_max = min(gray.shape[1], x_new + w_new + pad)
            y_max = min(gray.shape[0], y_new + h_new + pad)

            # Localized Translate
            sx_local = sx - x_min
            sy_local = sy - y_min

            # Isolate this specific token's stroke inside the local ROI (masking out neighbors)
            roi_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            local_contours = []
            for c in group_contours:
                local_c = c.copy()
                local_c[:, 0, 0] -= x_min
                local_c[:, 0, 1] -= y_min
                local_contours.append(local_c)

            cv2.drawContours(roi_mask, local_contours, -1, 255, thickness=-1)
            stroke_dilate = int(max(round(median_width * 0.5), 1))
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_dilate * 2 + 1, stroke_dilate * 2 + 1))
            roi_mask = cv2.dilate(roi_mask, kernel_dilate)

            # Crop from the SKELETON image (contains the 1-pixel centerline)
            actual_roi = skeleton[y_min:y_max, x_min:x_max]
            local_token = cv2.bitwise_and(actual_roi, roi_mask)

            # Affine De-rotation strictly inside local ROI matrix space (offset by -90.0 deg to align with UP-pointing templates)
            angle_deg = -float(theta * 180.0 / np.pi) - 90.0
            rot_matrix = cv2.getRotationMatrix2D((float(sx_local), float(sy_local)), angle_deg, 1.0)
            rotated_local = cv2.warpAffine(local_token, rot_matrix, (x_max - x_min, y_max - y_min), flags=cv2.INTER_LINEAR)

            # Crop tight bounding box of de-rotated token
            coords = np.argwhere(rotated_local > 0)
            if len(coords) == 0:
                cropped_token = rotated_local
            else:
                y_min_rot, x_min_rot = coords.min(axis=0)
                y_max_rot, x_max_rot = coords.max(axis=0)
                cropped_token = rotated_local[y_min_rot:y_max_rot+1, x_min_rot:x_max_rot+1]

            # Aspect-Ratio Preserved padding to square
            h_crop, w_crop = cropped_token.shape
            if h_crop > w_crop:
                pad_left = (h_crop - w_crop) // 2
                pad_right = (h_crop - w_crop) - pad_left
                padded_token = np.pad(cropped_token, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
            elif w_crop > h_crop:
                pad_top = (w_crop - h_crop) // 2
                pad_bottom = (w_crop - h_crop) - pad_top
                padded_token = np.pad(cropped_token, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_token = cropped_token.copy()

            # Downsample to exactly 28x28 using cv2.INTER_AREA & normalize float32 [0.0 - 1.0]
            resized_outward = cv2.resize(padded_token, (28, 28), interpolation=cv2.INTER_AREA)
            matrix_outward = resized_outward.astype(np.float32) / 255.0

            # Generate 180-degree flipped variant (inward-facing)
            resized_inward = cv2.flip(resized_outward, -1)
            matrix_inward = resized_inward.astype(np.float32) / 255.0

            token = NormalizedToken(
                token_id=idx_token,
                centroid=(int(round(sx)), int(round(sy))),
                placement_angle_rad=float(theta),
                matrix_outward=matrix_outward,
                matrix_inward=matrix_inward
            )
            tokens.append(token)

        # --- Debug Telemetry Overrides ---
        if debug_inspect:
            os.makedirs('./debug_output', exist_ok=True)
            # Create/overwrite visual diagnostics overlay 03_modifier_rois.png
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Draw parent ring contour in blue
            cv2.drawContours(debug_img, [parent_ring_contour], -1, (255, 0, 0), thickness=2)
            cx, cy = ring_metrics.centroid
            # Draw parent centroid in green
            cv2.circle(debug_img, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)

            # Renders tokens details
            for idx, (x_new, y_new, w_new, h_new, sx, sy) in enumerate(token_bboxes):
                token = tokens[idx]
                
                # 1. Green bounding box around clustered multi-stroke token
                cv2.rectangle(debug_img, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 255, 0), thickness=1)
                
                # 2. Red vector line connecting Ring Centroid to Token Centroid
                cv2.line(debug_img, (cx, cy), (int(round(sx)), int(round(sy))), (0, 0, 255), thickness=1)
                
                # 3. Text overlay displaying raw angle in radians rounded to 3 decimals
                label_text = f"{token.placement_angle_rad:.3f} rad"
                cv2.putText(debug_img, label_text, (x_new, max(y_new - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.imwrite('./debug_output/03_modifier_rois.png', debug_img)

        return tokens

    def _normalize_template(self, stroke_template: dict) -> np.ndarray:
        """
        Renders a vector stroke template onto a raster canvas and subjects it to the
        exact same preprocessing pipeline as the user's input, with morphological pre-smoothing.
        """
        # Create a black raster canvas
        canvas_sz = 200
        canvas = np.zeros((canvas_sz, canvas_sz), dtype=np.uint8)
        
        strokes = stroke_template.get("strokes", [])
        for stroke in strokes:
            if len(stroke) < 2:
                if len(stroke) == 1:
                    pt = stroke[0]
                    px = int(round(pt["x"] * canvas_sz))
                    py = int(round(pt["y"] * canvas_sz))
                    px = min(max(px, 0), canvas_sz - 1)
                    py = min(max(py, 0), canvas_sz - 1)
                    cv2.circle(canvas, (px, py), radius=3, color=255, thickness=-1)
                continue
                
            for k in range(len(stroke) - 1):
                pt1 = stroke[k]
                pt2 = stroke[k+1]
                p1x = int(round(pt1["x"] * canvas_sz))
                p1y = int(round(pt1["y"] * canvas_sz))
                p2x = int(round(pt2["x"] * canvas_sz))
                p2y = int(round(pt2["y"] * canvas_sz))
                
                p1x = min(max(p1x, 0), canvas_sz - 1)
                p1y = min(max(p1y, 0), canvas_sz - 1)
                p2x = min(max(p2x, 0), canvas_sz - 1)
                p2y = min(max(p2y, 0), canvas_sz - 1)
                
                cv2.line(canvas, (p1x, p1y), (p2x, p2y), 255, thickness=6)
                
        # Subject to identical pipeline as input
        # Binarize
        _, binary = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Flawless spur prevention guard: morphological open before skeletonization
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_smooth = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        # Skeletonize thinned centerline
        skeleton = self.zhang_suen_thinning(binary_smooth)
        
        # Clean vertex spurs using morphology open filter immediately after skeletonization
        # Dilate first to protect the thin 1-pixel centerline from erosion deletion
        skeleton_dilated = cv2.dilate(skeleton, kernel_open)
        skeleton_smoothed = cv2.morphologyEx(skeleton_dilated, cv2.MORPH_OPEN, kernel_open)
        # Re-skeletonize to yield a flawless 1-pixel wide canonical centerline path
        skeleton = self.zhang_suen_thinning(skeleton_smoothed)
        
        # Crop tight bounding box
        coords = np.argwhere(skeleton > 0)
        if len(coords) == 0:
            return np.zeros((28, 28), dtype=np.float32)
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = skeleton[y_min:y_max+1, x_min:x_max+1]
        
        # Aspect-Ratio Preserved padding to square
        h, w = cropped.shape
        if h > w:
            pad_left = (h - w) // 2
            pad_right = (h - w) - pad_left
            padded = np.pad(cropped, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        elif w > h:
            pad_top = (w - h) // 2
            pad_bottom = (w - h) - pad_top
            padded = np.pad(cropped, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
        else:
            padded = cropped.copy()
            
        # Downsample to exactly 28x28 using cv2.INTER_AREA
        resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize float32 range [0.0 - 1.0]
        return resized.astype(np.float32) / 255.0

    def compile_spell_diagram(
        self, 
        image_array: np.ndarray, 
        dictionary_paths: Dict[str, str]
    ) -> FinalSpellIR:
        """
        Executes the complete production pipeline (Steps 1 to 5).
        Loads dictionaries, isolates the activation parent ring, groups multi-stroke internal 
        tokens, resolves affine de-rotations, verifies orientation anchors via dual-metric 
        matching, and compiles the final validated Spell IR schema.
        """
        # 1. Dictionary Template Pre-execution Normalization & Caching
        templates = getattr(self, "preloaded_templates", {})
        if not templates:
            templates = {}
            for category, path in dictionary_paths.items():
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            class_id = item.get("id")
                            stroke_template = item.get("strokeTemplate")
                            if class_id and stroke_template:
                                # Pre-execution normalization caching
                                normalized = self._normalize_template(stroke_template)
                                templates[class_id] = normalized
                except Exception:
                    # Safe JSON load fallback
                    pass

        # 2. Isolate Parent Ring (Preprocessing & Skeletonization)
        parent_contour, ring_metrics = self.isolate_parent_ring(image_array, debug_inspect=False)
        
        # Short-circuit Integration Safety Guard
        if parent_contour is None or ring_metrics is None:
            return FinalSpellIR(
                is_valid_spell=False,
                ring_circularity=0.0,
                detected_symbols=[]
            )

        # 3. Leaf Symbol Spatial Extraction (Grouping & Centroid moments)
        tokens = self.extract_and_normalize_tokens(image_array, parent_contour, ring_metrics, debug_inspect=False)

        # 4. Step 5: Distance-Based Shape Matching
        detected_symbols = []
        
        # 9x9 dilation kernel to stabilize Hu Moments and eliminate division instability
        kernel_stabilize = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        
        for token in tokens:
            best_class = None
            best_hu_dist = float('inf')
            
            # Explicitly convert the 28x28 token matrix back into standard single-channel 8-bit unsigned integer mask (uint8)
            token_u8 = (token.matrix_outward * 255.0).astype(np.uint8)
            # Apply low threshold to binarize cleanly
            _, token_bin = cv2.threshold(token_u8, 1, 255, cv2.THRESH_BINARY)
            # Dilate to give the 1-pixel centerline stable contour boundaries
            token_dil = cv2.dilate(token_bin, kernel_stabilize)
            
            # Extract primary structural boundaries
            contours_token, _ = cv2.findContours(token_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c_token = max(contours_token, key=cv2.contourArea) if contours_token else None
            
            # Find closest structural class using cv2.matchShapes (Hu Moments method)
            for class_id, ref_template in templates.items():
                # Convert the cached reference template matrix back into standard single-channel 8-bit unsigned integer mask (uint8)
                ref_u8 = (ref_template * 255.0).astype(np.uint8)
                _, ref_bin = cv2.threshold(ref_u8, 1, 255, cv2.THRESH_BINARY)
                ref_dil = cv2.dilate(ref_bin, kernel_stabilize)
                
                # Extract primary structural boundaries
                contours_ref, _ = cv2.findContours(ref_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c_ref = max(contours_ref, key=cv2.contourArea) if contours_ref else None
                
                if c_token is not None and c_ref is not None:
                    dist = float(cv2.matchShapes(c_token, c_ref, cv2.CONTOURS_MATCH_I1, 0.0))
                else:
                    dist = 999.0 # Fallback high distance if contours are missing
                    
                if dist < best_hu_dist:
                    best_hu_dist = dist
                    best_class = class_id

            if best_class is not None:
                # 5. Orientation Resolution & shape similarity via binarized L2 MSE to eliminate sparse-skeleton bias
                template_matrix = templates[best_class]
                
                token_bin_f32 = (token.matrix_outward > 0.05).astype(np.float32)
                template_bin_f32 = (template_matrix > 0.05).astype(np.float32)
                token_inward_bin_f32 = (token.matrix_inward > 0.05).astype(np.float32)
                
                mse_outward = float(np.mean((token_bin_f32 - template_bin_f32) ** 2))
                mse_inward = float(np.mean((token_inward_bin_f32 - template_bin_f32) ** 2))
                
                if mse_outward <= mse_inward:
                    orientation = "outward"
                    min_mse = mse_outward
                else:
                    orientation = "inward"
                    min_mse = mse_inward
                
                # Combined Multiplicative Confidence Score Mapping
                confidence = (1.0 - min_mse) * (1.0 / (1.0 + best_hu_dist))
                
                # Absolute Rejection Threshold (Floor = 0.70)
                if confidence < 0.70:
                    symbol_class = "unknown"
                else:
                    symbol_class = best_class
            else:
                symbol_class = "unknown"
                confidence = 0.0
                orientation = "outward"

            detected_symbols.append(ParsedSymbol(
                symbol_class=symbol_class,
                confidence_score=confidence,
                placement_angle_rad=token.placement_angle_rad,
                orientation=orientation
            ))

        return FinalSpellIR(
            is_valid_spell=ring_metrics.is_closed,
            ring_circularity=ring_metrics.circularity_score,
            detected_symbols=detected_symbols
        )
