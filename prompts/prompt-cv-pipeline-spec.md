## Project Component: Computer Vision Pipeline for WHAtelier Spell Simulator

### 1. Objective
Build a robust, high-performance backend image preprocessing and object segmentation pipeline using `opencv-python` and `numpy`. The pipeline must deterministically isolate hand-drawn magic circles ("Rings") and extract their internal modifier tokens ("Sigils" and "Signs") from a live canvas coordinate stream or rasterized image, ensuring clean input vectorization for downstream machine learning inference.

### 2. Theoretical Architecture & Hierarchy
The system maps visual syntax topology using an explicit **Contour Hierarchy** tree (`cv2.RETR_TREE`, `cv2.CHAIN_APPROX_SIMPLE`). 
* **Topological Root (Level 0):** The Activation Ring. Identified via geometric constraints (maximum area enclosed among high-circularity candidates).
* **Leaf Nodes (Level 1):** Semantic modifier symbols enclosed *strictly* within the spatial boundary of the Root Node.
* **Pipeline Output:** A clean, decoupled data structure consisting of the Activation Ring state metrics and an array of normalized $28 \times 28$ matrix regions of interest (ROIs) for the modifiers.

---

### 3. Algorithmic Resolutions for Edge Cases (The Technical Blueprint)

To bypass the brittle assumptions of standard geometric computer vision, the implementation must execute the following algorithmic mitigations sequentially:

```
[Input Image] 
      │
      ▼
[Phase A: Gap-Bridging Engine] ──► Resolves Unclosed Rings
      │
      ▼
[Phase B: Junction-Slicing Engine] ──► Resolves Intersecting/Attached Signs
      │
      ▼
[Contour Hierarchy Extraction] 
      │
      ▼
[Phase C: Spatial Agglomeration] ──► Resolves Multi-Stroke/Fragmented Signs
      │
      ▼
[Phase D: ROI Vectorizer] ──► Generates Normalised 28x28 Matrices
```

#### A. The Unclosed Ring / Boundary Leakage (Gap-Bridging Engine)
* **Problem:** Manual brush strokes often leave a minor gap ($\Delta\text{gap}$). Standard contour retrieval treats this as an open line rather than a closed parent container, causing internal symbols to leak into the global background hierarchy level.
* **Mathematical Strategy:** Implement an adaptive endpoint pairing algorithm rather than standard global morphological dilation (which distorts fine features).
* **Algorithmic Execution Steps:**
  1. Apply topological skeletonization (e.g., Guo-Hall or Zhang-Suen algorithm) to reduce the binary mask to 1-pixel width.
  2. Locate line endpoints by convolving the skeleton with a $3 \times 3$ neighborhood filter. A pixel $p$ is an endpoint if its 8-connected neighborhood sum equals 2 (the pixel itself plus exactly one neighbor):
     $$\sum_{i=1}^{8} N(p)_i = 1$$
  3. Compute a local pairwise Euclidean distance matrix $D$ between all detected endpoints:
     $$d(P_i, P_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$
  4. For any pairs where $d(P_i, P_j) \le \tau_{\text{gap}}$, programmatically draw a direct connecting stroke using `cv2.line` with thickness matching the estimated median stroke width of the canvas.

#### B. Connected Contours (Junction-Slicing Engine)
* **Problem:** When a modifier sign intersects or physically touches the perimeter of the Activation Ring, OpenCV merges them into a single continuous contour polygon. This completely destroys the `Parent-Child` hierarchical topology.
* **Mathematical Strategy:** Isolate the intersecting junctions using a combination of **Euclidean Distance Transform (EDT)** and **Topological Valence Analysis**.
* **Algorithmic Execution Steps:**
  1. Calculate the L2 Distance Transform (`cv2.distanceTransform`) of the merged binary image. This computes the distance from each foreground pixel to the nearest background pixel.
  2. Isolate the skeleton of the intersection area. Map the valence (degree) of each pixel vertex in the skeleton graph.
  3. Identify junction nodes where the vertex degree is $\ge 3$ (indicating a branch point where a sign meets the ring).
  4. Apply a localized masking operation: For every junction node coordinate, draw a black isolation circle with a radius proportional to the local distance transform value $\tau_{\text{slice}} = \text{EDT}(x, y) \cdot 1.2$. This cleanly slices the sign away from the ring at the exact point of contact while preserving geometric shapes.

#### C. Fragmented Multi-Stroke Modifiers (Spatial Agglomeration Engine)
* **Problem:** Complex signs drawn using multiple discrete strokes (e.g., multi-line glyphs, accents, dots) register as disconnected individual child contours, causing a single semantic token to split into multiple arbitrary payloads.
* **Mathematical Strategy:** Apply **Hierarchical Agglomerative Clustering (HAC)** based on bounding-box proximity metrics.
* **Algorithmic Execution Steps:**
  1. Extract all raw internal child contours enclosed by the parent ring, obtaining their bounding slices:
     $$B_i = [x_i, y_i, w_i, h_i]$$
  2. Define the spatial distance metric between two bounding boxes $B_A$ and $B_B$ as the minimum clearance distance between their edges along both axes.
  3. Construct a proximity graph. If the minimum distance $Dist(B_A, B_B) \le \tau_{\text{group}}$ (where $\tau_{\text{group}}$ is dynamically set to $1.5 \times \text{median stroke width}$), create an edge between Node $A$ and Node $B$.
  4. Find the Connected Components of this proximity graph. For each component cluster, merge the sub-bounding boxes into a unified bounding box:
     $$x_{\text{new}} = \min(x_A, x_B), \quad y_{\text{new}} = \min(y_A, y_B)$$
     $$w_{\text{new}} = \max(x_A + w_A, x_B + w_B) - x_{\text{new}}$$
     $$h_{\text{new}} = \max(y_A + h_A, y_B + h_B) - y_{\text{new}}$$

---

### 4. Technical Constraints & Execution Specifications

* **Environment Environment:** Python 3.11+, NumPy $\ge$ 1.24, OpenCV-Python $\ge$ 4.8.
* **Determinism:** Absolutely zero hardcoded pixel constants. All operational tolerances ($\tau_{\text{gap}}$, $\tau_{\text{slice}}$, $\tau_{\text{group}}$) must be calculated dynamically relative to the image dimensions or the median stroke width.
* **Activation Ring Validation Metric:** To prevent small internal details or stray background marks from spoofing the Activation Ring, candidates must satisfy the circularity coefficient threshold $S$:
  $$S = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2} \ge \tau_{\text{circularity}} \quad (\text{Default } \tau_{\text{circularity}} = 0.72)$$

* **Phase D: ROI Normalization Requirements:**
  Before passing any extracted bounding box to the vectorizer array, the sub-matrix must be transformed cleanly:
  1. **Aspect Ratio Preservation:** Calculate the aspect ratio of the unified child bounding box. Pad the shorter axis symmetrically with background pixels (0) to create a perfect square matrix. Do not warp or stretch the image directly.
  2. **Interpolation:** Downsample the square matrix to exactly $28 \times 28$ pixels using `cv2.INTER_AREA` (optimal for decimation/shrinking).
  3. **Min-Max Scaling:** Convert the datatype to `float32` and normalize pixel values linearly to the scale $[0.0, 1.0]$.