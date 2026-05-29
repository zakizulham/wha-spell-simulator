# Changes Log: Steps 3 & 4 (Leaf Symbol Extraction & De-rotation Normalization)

All files created, configured, and tested for Project #1 (Contour Matching Engine - Steps 3 & 4) in this session are tracked below.

---

## 1. Added & Modified Components

### A. Core Engine Implementation
#### [MODIFY] [spell_parser.py](file:///c:/Users/ZakiZ/Documents/01_projects/wha-spell-simulator/spell_parser.py)
* **Added Data Structures:** Incorporated the `@dataclass NormalizedToken` containing the unique ID, definitive centroid coordinates, placement angle in radians, and the dual normalized $28 \times 28$ matrices (`matrix_outward` and `matrix_inward`).
* **Adaptive Enclosure Filtering (Step 3):** Added `cv2.pointPolygonTest` with an adaptive tolerance threshold mapped to the median stroke width ($W_m$) to robustly include symbols that slightly overlap the wavy parent ring border:
  $$\text{is\_inside} \iff \text{pointPolygonTest}(ParentContour, (S_x, S_y), \text{True}) \ge - (W_m \times 0.5)$$
* **Stroke Proximity Grouping (Step 3):** Implemented a Hierarchical Agglomerative Clustering (HAC) graph search. If the minimum 2D clearance distance between any two child bounding boxes is $\le 1.5 \times W_m$, they are clustered into a unified token.
* **Definitive Centroid Calculation (Step 3):** Spatial moments are calculated on the concatenated contours of the unified group:
  $$S_x = \frac{m_{10}}{m_{00}}, \quad S_y = \frac{m_{01}}{m_{00}}$$
* **Displacement Angle Topology (Step 4):** Calculates the radial displacement vector:
  $$\theta = \text{atan2}(S_y - C_y, S_x - C_x)$$
* **Localized Padded Affine De-rotation (Step 4 Override):** Implemented a localized cropping workflow to prevent cropping out-of-bounds:
  1. Crop the ROI from the canvas using a bounding box padded by $2 \times W_m$ on all four sides.
  2. Translate global coordinates to local space: $S_{x\_local} = S_x - x_{min}$.
  3. Generate a clean token mask inside the ROI, dilate it by $0.5 \times W_m$ to fully capture ink strokes, and mask out neighboring drawings.
  4. Execute rotation warp (`cv2.warpAffine`) by $-\theta$ around the local centroid.
  5. Tight crop the de-rotated result.
* **Standardized 28x28 Normalization (Step 4):** Symmetrically zero-pads the shorter axis to form a perfect square, downsamples via `cv2.INTER_AREA` to exactly $28 \times 28$ pixels, casts to `float32`, scales linearly to $[0.0, 1.0]$, and produces outward and inward (flipped by 180 degrees) variants.
* **Diagnostics Visual telemetries overlay:** Overwrites/appends overlay metrics to `./debug_output/03_modifier_rois.png`:
  1. Green bounding boxes around unified, clustered multi-stroke tokens.
  2. A red vector line connecting the Parent Centroid ($C_x, C_y$) to each Token Centroid ($S_x, S_y$).
  3. A red text overlay displaying raw placement angle $\theta$ in radians rounded to 3 decimal places.

### B. Verification Test Suite
#### [MODIFY] [test_spell_parser.py](file:///c:/Users/ZakiZ/Documents/01_projects/wha-spell-simulator/tests/test_spell_parser.py)
* **Added Test Case:** `test_leaf_symbol_spatial_agglomeration_and_normalization`
  - Simulates a complete, multi-stroke symbol (a line and a dot drawn with a physical gap) inside a parent ring, oriented at a 45-degree angle.
  - Verifies that HAC successfully groups the disconnected strokes into exactly 1 unified token.
  - Asserts that centroid, displacement angle ($\approx 0.785$ rad), $28 \times 28$ shapes, float32 type, and $[0.0, 1.0]$ ranges are 100% correct.
  - Confirms the correct generation of debug telemetries under `./debug_output/03_modifier_rois.png`.

---

## 2. Execution & Verification Status
* **Test Command:** `.\.venv\Scripts\python -m unittest tests/test_spell_parser.py`
* **Test Results:** 6/6 unit tests passed successfully in 0.185s with zero errors or failures.
* **Visual Telemetry:** Fresh telemetry images with green rectangles, red lines, and radian labels generated under `./debug_output/03_modifier_rois.png`.
