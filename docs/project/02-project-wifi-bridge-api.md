# Project Specifications: Project #2

## 1. Project Metadata
* **Title:** Low-Latency Wi-Fi Bridge API Engine
* **Subtitle:** FastAPI Network Router for Cross-Platform Canvas Synchronization
* **Goal Project:** Create an ultra-lightweight, high-performance HTTP REST/Websocket API server using Python and FastAPI. This server bridges the Android tablet's frontend drawing canvas with the `DeterministicSpellParser` backend over the local Wi-Fi network with sub-10ms transmission latency.

---

## 2. Description
To achieve a seamless interactive experience using a stylus pen on an Android tablet, the architecture must decouple the frontend canvas from the computer vision parser. 

Project #2 implements `app.py`, a localized API server. It listens on all local network interfaces (`0.0.0.0`), bypasses browser security blocks by enabling full CORS (Cross-Origin Resource Sharing), accepts incoming binary canvas frames or coordinate arrays from the tablet, executes the complete Step 1–5 pipeline via Project #1's engine, and streams back the compiled `FinalSpellIR` data structure.

---

## 3. Expected Input & Output Contracts

### Network Input (POST /api/parse-spell)
The Android tablet browser will send a JSON payload containing the drawn canvas data as a Base64 encoded PNG string:
```json
{
    "session_id": "witch_player_1",
    "timestamp": 1716942100,
    "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Network Output (JSON Response)
The server returns the exact compiled Spell IR schema from Project #1 alongside server telemetry:
```json
{
    "status": "success",
    "parse_time_ms": 14.2,
    "payload": {
        "is_valid_spell": true,
        "ring_circularity": 0.91,
        "detected_symbols": [
            {
                "symbol_class": "fire_sigil",
                "confidence_score": 0.88,
                "placement_angle_rad": 0.0,
                "orientation": "outward"
            }
        ]
    }
}
```

---

## 4. Architectural Steps (Implementation Pipeline)

```
[Android Stylus Input] ──► (Wi-Fi POST Request) ──► [FastAPI Router (0.0.0.0)]
                                                           │
                                                           ▼
                                                [CORS / Security Clearance]
                                                           │
                                                           ▼
                                                [Base64 Image Decoder]
                                                           │
                                                           ▼
                                                [Project #1 Math Engine]
                                                           │
                                                           ▼
[Visual Spell Effects] ◄── (JSON Response Contract) ◄── [Data Schema Packager]
```

### Step 1: Framework Setup & CORS Injection
* Instantiated server via `FastAPI`.
* Inject `CORSMiddleware` to allow unrestricted access from `allow_origins=["*"]`, `allow_methods=["*"]`, and `allow_headers=["*"]`. This prevents Android Chrome from dropping requests due to same-origin security policies.

### Step 2: Base64 Decryption Pipeline
* Read the incoming `image_base64` string.
* Strip the header metadata (`data:image/png;base64,`).
* Decode the raw string back into a binary buffer using Python's `base64.b64decode`.
* Convert the binary buffer into a NumPy single-channel or RGB matrix array using `cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)`.

### Step 3: Engine Execution & Benchmark
* Pass the generated matrix directly into `DeterministicSpellParser.compile_spell_diagram()`.
* Wrap the execution inside a microsecond high-resolution timer (`time.perf_counter()`) to profile latency analytics.

---

## 5. Detailed Execution Constraints (Rules of Implementation)

* **The Host Binding Rule:** The server must NOT run on `127.0.0.1` or `localhost`. It must be bound explicitly to `0.0.0.0` at port `8000`. This forces Uvicorn to listen to the machine’s wireless network card, making it discoverable by the tablet.
* **The Dependency Isolation Rule:** Add `fastapi`, `uvicorn`, and `pydantic` to the `requirements.txt` file. Do not introduce bloated web frameworks (like Django). Keep memory footprints under 50MB.
* **Fail-Safe Contract Exception:** If the Base64 string is corrupted or unreadable, the API must return an HTTP 400 Bad Request payload with a clear structural error message: `{"status": "error", "message": "Invalid Base64 byte-stream syntax"}`.