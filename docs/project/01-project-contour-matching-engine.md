# Project Specifications: Project #1

## 1. Project Metadata
* **Title:** Deterministic Contour-Driven Spell Recognition Engine
* **Subtitle:** Pure Mathematics and Geometric Topology Parsing for Real-Time Canvas Input
* **Goal Project:** Build an ultra-reliable, zero-latency computer vision backend that translates freehand stylus contours into deterministic gameplay parameters (Spell IR). It must eliminate the statistical instability, alignment issues, and stroke-thickness sensitivity inherent in pixel-overlap and machine learning approaches.

---

## 2. Description
In game engineering, mechanics must feel fair and predictable. The original raster-based "pixel overlap counting" methodology fails under real-world human drawing variance (such as slight translations, varying stylus brush widths, and hand tremors). 

This project re-engineers the symbol recognition layer by treating the canvas drawings as pure continuous vector contours. By shifting from pixel matrix density to topological features—specifically utilizing Spatial/Central Moments for alignment and Shape Distance Matchers—the engine achieves invariance against scale, rotation, and translation. Furthermore, it converts mathematical error margins (deviation scores) into a continuous float scale to drive in-game mechanics like spell volatility, stability, and area-of-effect scaling.

---

## 3. Expected Output
The algorithm must return a clean, decoupled Python data structure (Dictionary or Dataclass) representing the Intermediate Representation (Spell IR):

```python
{
    "ring_integrity": {
        "is_closed": True,
        "circularity_score": 0.89,     # Float [0.0 - 1.0]
        "radius_px": 245.5,
        "centroid": (500, 420)         # (X, Y) coordinate
    },
    "detected_modifiers": [
        {
            "symbol_class": "fire_sigil",
            "confidence_score": 0.94,  # Computed from shape distance inversed
            "relative_angle_rad": 0.0, # Rotation relative to center
            "scale_factor": 1.15       # Physical size compared to template
        },
        {
            "symbol_class": "column_sign",
            "confidence_score": 0.81,
            "relative_angle_rad": 1.57,# 90 degrees (facing right)
            "scale_factor": 0.85
        }
    ]
}
```

---

## 4. Algoritma (Step-by-Step Pipeline)

```
 [Raw Canvas Input] ──► [Binarization & Skeletonization]
                              │
                              ▼
                [Contour Extraction & Filtering]
                              │
                              ▼
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
 [Ring Analysis]                           [Modifier Separation]
  - Circularity Metric                      - Calculate Central Moments (Centroid)
  - Convex Hull Distance                    - Radial Angle Calculation
                                            - De-rotation & Scaling Matrix
                                                   │
                                                   ▼
                                         [Shape Matching Metric]
                                          - Invariant Hu Moments
                                          - Contour Distance Matrix
```

### Step 1: Preprocessing & Skeletonization
1. Convert the canvas source image to a grayscale single-channel array.
2. Apply Otsu's adaptive thresholding to produce a clean binary mask (0 for background, 255 for stroke).
3. Apply a skeletonization algorithm to reduce the stroke thickness to a 1-pixel-wide line path, removing stroke-width bias.

### Step 2: Parent Ring Isolation
1. Extract contours using hierarchical tracking. Find the outer-most closed loop candidate.
2. Calculate the area (A) and perimeter (P) of the candidate loop.
3. Compute Circularity: S = (4 * pi * Area) / (Perimeter^2).
4. If S >= Threshold_Circularity, flag this contour as the Active Magic Ring and retrieve its Centroid (Cx, Cy).

### Step 3: Leaf Symbol Spatial Extraction
1. Filter all contours whose bounding boxes exist strictly inside the boundaries of the Parent Ring.
2. For each disconnected child contour, calculate its center of mass using spatial moments:
   X_center = m10 / m00
   Y_center = m01 / m00

### Step 4: Normalization (Scale & Rotation Invariance)
1. Calculate the spatial vector from the Ring Centroid (Cx, Cy) to the Symbol Centroid (Sx, Sy).
2. Determine the orientation angle theta = arctan2(Sy - Cy, Sx - Cx).
3. Apply an affine transformation matrix to rotate the symbol contour back by -theta, aligning it perfectly to the baseline template orientation (facing down/0-degrees).
4. Resize the contour bounding box dimensions to match the template bounding scale.

### Step 5: Distance-Based Shape Matching
1. Compute the structural similarity score between the normalized child contour and known dictionary templates using Hu Moments distance metrics.
2. Inverse the metric value to represent a 0.0 to 1.0 confidence score (where 1.0 is a flawless geometric match).

---

## 5. Detailed Execution Constraints (Rules of Implementation)

* **Anti-Machine-Number Rule:** Do not use static pixel values for distance checks. Every operational threshold must be calculated dynamically. For example, the distance threshold for merging fragmented strokes must be written as:
  `T_group = median_stroke_width * 1.5`

* **Circular Leakage Verification:** Instead of using pixel fill routines to check for unclosed rings, calculate the distance between the starting point and ending point of the Ring's open contour vector. If `distance(start, end) > median_stroke_width * 2`, the ring is mathematically open.

* **Shape Matcher Metric Specification:** Use OpenCV's Shape Matcher with the log-analytic Hu Moments method formula:
  `I_1(A,B) = sum( abs( 1/eta_i^A - 1/eta_i^B ) )`
  This ensures that minor hand tremors or wiggly paths do not massively penalize the score, unlike pixel overlap checks.

* **Continuous Balancing Map:** Map the final similarity score to the gameplay volatility scale. 
  * If score > 0.85: Spell Stability = 100% (Perfect execution).
  * If 0.70 <= score <= 0.85: Spell Stability = score * 100 (Volatile execution, add particle dispersion).
  * If score < 0.70: Fail execution (`Spell Fizzled`).