# Changes Log: Project #2 (FastAPI Low-Latency Wi-Fi Bridge API Engine)

This session successfully designed, built, optimized, and integrated the high-performance **FastAPI Wi-Fi Bridge API** (`app.py`), bridging frontend canvas inputs with our deterministic OpenCV spell parser engine.

---

## 1. Added & Modified Components

### A. Core Wi-Fi Bridge Implementation
#### [NEW] [app.py](file:///c:/Users/ZakiZ/Documents/01_projects/wha-spell-simulator/app.py)
* **ASGI Event-Loop Thread Safeguard:** Declared the route strictly as a standard synchronous function (`def parse_spell(request: SpellParseRequest):` instead of `async def`) to automatically offload heavy CPU-bound OpenCV matrix parsing and skeleton calculations onto FastAPI's internal background worker thread pool.
* **CORS Mobile Discovery Clearance:** Configured `CORSMiddleware` with all origins (`["*"]`), methods, and headers allowed, with credential cookies disabled to avoid browser engine startup exceptions, enabling seamless Android tablet stylus canvas connections.
* **Boot-Time Pre-Caching Lifecycle:** Registered an `@app.on_event("startup")` hook that parses reference template files (`sigils.json` and `signs.json`) and caches their vector contours in-memory at boot, avoiding filesystem latency on live game request threads.
* **Post-Decode Structural Array Validation:** Injected strict binary matrix checks after Base64 decoding and image decoding:
  ```python
  if decoded_matrix is None or decoded_matrix.size == 0:
      raise HTTPException(status_code=400, detail="Invalid Base64 byte-stream syntax: Decoded image matrix is empty")
  ```

### B. Core Engine Cache Hook
#### [MODIFY] [spell_parser.py](file:///c:/Users/ZakiZ/Documents/01_projects/wha-spell-simulator/spell_parser.py)
* **Zero-Disk-Latency Hook:** Initialized the `self.preloaded_templates` attribute and modified `compile_spell_diagram` to check this in-memory cache first, entirely bypassing expensive local template disk reads and rasterization thinners for active requests.

### C. Dependency Setup
#### [MODIFY] [requirements.txt](file:///c:/Users/ZakiZ/Documents/01_projects/wha-spell-simulator/requirements.txt)
* Locked PyPI precompiled wheel dependencies for Python 3.14.2 compatibility:
  ```txt
  fastapi==0.136.3
  uvicorn==0.48.0
  pydantic==2.13.4
  ```

### D. Automated Integration Testing
#### [NEW] [tests/test_api_bridge.py](file:///c:/Users/ZakiZ/Documents/01_projects/wha-spell-simulator/tests/test_api_bridge.py)
* Developed full coverage tests using FastAPI `TestClient` for:
  1. Successful parsing of programmatically generated synthetic spell drawings.
  2. Malformed/invalid base64 syntax handling.
  3. Valid base64 strings decoding to empty/corrupted image buffers.
  4. CORS header injection check for OPTIONS preflight and GET/POST methods.

---

## 2. Execution & Regression Verification Status

* **Test Command:** `.\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"`
* **Test Results:** 12/12 unit and integration tests passed in 0.913s with zero failures.

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
C:\Users\ZakiZ\Documents\01_projects\wha-spell-simulator\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa
............
----------------------------------------------------------------------
Ran 12 tests in 0.913s

OK
```
