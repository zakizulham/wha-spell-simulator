# Witch Hat Atelier Spell Simulator

A fan-made browser-based spell drawing simulator inspired by *[Witch Hat Atelier](https://en.wikipedia.org/wiki/Witch_Hat_Atelier)*.

<div align="center">
  <img src="./assets/demo.gif" width="720"/>
  <p>Try here: <a href="https://ytnrvdf.github.io/wha-spell-simulator">https://ytnrvdf.github.io/wha-spell-simulator</a></p>
</div>

## Fan Project Notice

This is an unofficial fan-made project for learning, experimentation, and appreciation. It is not affiliated with, endorsed by, or sponsored by the official creators, publishers, licensors, or production partners of *Witch Hat Atelier*.

*Witch Hat Atelier* and related names, artwork, symbols, and trademarks belong to their respective rights holders. The sigils, signs, spell terminology, and visual effects in this project are partial fan references and interactive interpretations, not official assets or canonical rules.

## What It Does

The app turns a freehand spell diagram into parser output, compiled spell behavior, and animated canvas effects.

- Lets you draw spell diagrams on a paper-like canvas.
- Detects one enclosing ring and distinguishes prepared versus active spells.
- Recognizes dictionary-backed primary sigils for fire, water, wind, earth, and light.
- Recognizes signs that modify direction, levitation, convergence, force, spread, focus, range, duration, and stability.
- Produces parser diagnostics, `GlyphAST`, and `SpellIR` output for inspection.
- Renders animated element effects from the compiled spell behavior.
- Shows sample spell layouts in the Dictionary panel as drawing references.
- Includes reference tools for making, viewing, and testing stroke templates, plus a spell effect lab for visual and animation tuning.

---

## The Python Deterministic Math Engine (Fork Contribution)

This fork introduces a **production-grade computer vision backend written in Python 3.14.2**, replacing the legacy JavaScript raster-based pixel-counting parser. By treating freehand stylus coordinates as topological vector paths, it provides high-precision, zero-latency, and deterministic spell recognition tailored for tablet/stylus input.

### Key Architectural Enhancements:
1. **Vectorized Zhang-Suen Thinning:** Implemented pure NumPy array slicing to reduce arbitrary stroke widths down to flawless 1-pixel centerlines in $O(N)$ iteration time, removing native loop performance bottlenecks.
2. **Convolved Endpoint & Gap-Bridging Engine:** Uses a non-saturating `float32` 3x3 convolution filter to dynamically pair and heal unclosed ring leaks within a proximity threshold of $\tau_{\text{gap}} = 4.5 \times W_m$.
3. **Concentric Loop Resolution:** Gracefully identifies nested rings via `cv2.findContours(RETR_TREE)` and automatically maps the maximum area contour as the master Parent Activation Ring.
4. **Adaptive Boundary Tolerance:** Employs an adaptive `cv2.pointPolygonTest` boundary buffer scaled to the calculated median stroke width ($W_m$), successfully keeping wavy stylus lines that bleed slightly out of the ring edge from being dropped.
5. **Hierarchical Agglomerative Clustering (HAC):** Automatically groups multi-stroke fragments (e.g., disconnected sigil vectors) into a single unified token based on dynamic 2D bounding box clearance checks.
6. **Localized Affine De-rotation:** Resolves the exact placement angle $\theta$ using spatial moments and `atan2`, automatically warping the token ROI back to a 0-degree baseline without global clipping risks.
7. **Dual-Metric Disambiguation:** Fuses **Hu Moments** and **Binarized L2 Mean Squared Error (MSE)** matrices to perfectly distinguish orientation anchors (`oriented_outward` vs. `oriented_inward`), fully bringing the 180-degree directional canvas mechanics from the manga to life while cleanly rejecting chaotic scribbles as `"unknown"`.

---

## Current Limitations

- The app supports one enclosing spell ring at a time. Multiple rings are detected as unsupported.
- The current compiler expects one primary sigil. Multiple primary sigils are detected as unsupported.
- The dictionaries only cover a small fan-made subset of sigils, signs, and observed spell ideas.
- The visual effects are interpretive canvas animations, not a faithful reproduction of manga or anime effects.
- Raster images can be used as visual references, but the app cannot recover true stroke order from an image.
- Closed but invalid diagrams may show diagnostics, but they do not fall back to another element.
- This is a browser prototype, not a production drawing engine or general symbol recognizer.

---

## Setup & Running Locally

This repository operates as a *hybrid stack* (Frontend + Computer Vision Engine). Ensure your environment is running **Node.js** and **Python 3.14.2**.

### 1. Frontend Development Server (Node.js / Vite)

Install frontend dependencies:
```sh
npm install
```

Start the Vite dev server with network-broadcasting enabled for external tablet testing:
```sh
npm start -- --host
```

Then open the network IP or local address listed in your terminal:
```txt
[http://127.0.0.1:5173/](http://127.0.0.1:5173/)
```

### 2. Math Parser Engine (Python 3.14+ / PowerShell Workflow)

Instantiate and isolate the local Python environment using Windows PowerShell:
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install core mathematical dependencies
pip install -r requirements.txt
```

---

## Reference Tools

These tools are available from the app:

```txt
/tools/strokeTemplateMaker.html
/tools/strokeTemplateViewer.html
/tools/sigilSignDetectorLab.html
/tools/spellEffectLab.html
```

---

## Tests

### Node.js Frontend Tests
Run the legacy Node test suite:
```sh
npm test
```

### Python Deterministic Parser Tests
Run the robust OpenCV/NumPy test matrix covering skeletonization, gap bridging, HAC clustering, and dual-metric orientation disambiguation:
```powershell
.\.venv\Scripts\python -m unittest tests/test_spell_parser.py
```

---

## Documentation & Spec Sheets

- [01: Deterministic Contour-Driven Spec Engine](docs/projects/01-project-contour-matching-engine.md)
- [Dictionary authoring](docs/dictionary-authoring.md)
- [Parser and spell semantics rules](docs/play-rules.md)
- [Parsed glyph output contract](docs/glyph-ast.md)
- [Compiled spell output contract](docs/spell-ir.md)
- [Visual effect renderer notes](docs/effect-rendering.md)
