import base64
import time
import os
import json
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from spell_parser import DeterministicSpellParser

app = FastAPI(title="WHA Spell Simulator Wi-Fi Bridge")

# Inject CORSMiddleware configuration (Allow all origins, methods, and headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins matches ["*"] to prevent FastAPI runtime errors
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiation of our Project #1 Parser
parser = DeterministicSpellParser()
DICTIONARY_PATHS = {
    "sigils": "src/dictionary/sigils.json",
    "signs": "src/dictionary/signs.json"
}

class SpellParseRequest(BaseModel):
    session_id: str
    image_base64: str

@app.on_event("startup")
def preload_dictionaries():
    """Cache and normalize vector JSON shapes on server boot to secure zero-latency lookups."""
    preloaded = {}
    for category, path in DICTIONARY_PATHS.items():
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
                        normalized = parser._normalize_template(stroke_template)
                        preloaded[class_id] = normalized
        except Exception as e:
            print(f"Failed to preload {category} templates: {e}")
    parser.preloaded_templates = preloaded

@app.post("/api/parse-spell")
def parse_spell(request: SpellParseRequest):
    """
    Synchronously parses base64 magic spell drawing canvas images.
    Enforcing 'def' (omitting 'async') offloads the CPU-bound OpenCV parsing calculations
    to FastAPI's background worker threads, preventing event loop starvation.
    """
    try:
        start_time = time.perf_counter()
        
        # 1. Strip base64 header metadata if present and decode base64
        b64_data = request.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
            
        try:
            decoded_bytes = base64.b64decode(b64_data)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Base64 byte-stream syntax: {str(e)}"
            )
            
        # 2. Convert base64 bytes to NumPy buffer and decode using OpenCV
        np_buffer = np.frombuffer(decoded_bytes, dtype=np.uint8)
        decoded_matrix = cv2.imdecode(np_buffer, cv2.IMREAD_UNCHANGED)
        
        # Post-Decode Structural Array Validation (prevent silent failures/crashes)
        if decoded_matrix is None or decoded_matrix.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid Base64 byte-stream syntax: Decoded image matrix is empty"
            )
            
        # 3. Execute parser's end-to-end compilation pipeline
        final_ir = parser.compile_spell_diagram(decoded_matrix, DICTIONARY_PATHS)
        
        # 4. Compute elapsed time
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        # 5. Package and return the spell payload and performance telemetry
        return {
            "status": "success",
            "parse_time_ms": round(elapsed_time_ms, 2),
            "payload": {
                "is_valid_spell": final_ir.is_valid_spell,
                "ring_circularity": round(final_ir.ring_circularity, 4),
                "detected_symbols": [
                    {
                        "symbol_class": symbol.symbol_class,
                        "confidence_score": round(symbol.confidence_score, 4),
                        "placement_angle_rad": round(symbol.placement_angle_rad, 4),
                        "orientation": symbol.orientation
                    }
                    for symbol in final_ir.detected_symbols
                ]
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Parsing Failure: {str(e)}"
        )

if __name__ == "__main__":
    HOST = os.getenv("WHA_API_HOST", "127.0.0.1")
    PORT = int(os.getenv("WHA_API_PORT", 8000))
    
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)