import base64
import cv2
import numpy as np
import unittest
from fastapi.testclient import TestClient
from app import app

class TestAPIBridge(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_parse_spell_success(self):
        """
        Verifies that a valid base64 image containing a closed magic ring
        is successfully processed, returning status 200, positive latency profiling,
        and high circularity scores in the structured JSON payload.
        """
        # Generate a synthetic closed circle image (representing a valid spell ring)
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 60, 255, thickness=6)
        
        # Encode the OpenCV image matrix into a PNG buffer
        success, buffer = cv2.imencode(".png", img)
        self.assertTrue(success, "Failed to encode synthetic test image to PNG")
        
        # Base64-encode the PNG buffer
        b64_string = base64.b64encode(buffer).decode("utf-8")
        payload_b64 = f"data:image/png;base64,{b64_string}"
        
        # Construct and dispatch the JSON request contract
        request_body = {
            "session_id": "integration_test_session_99",
            "image_base64": payload_b64
        }
        
        response = self.client.post("/api/parse-spell", json=request_body)
        
        # Assertions to verify the contract requirements
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data.get("status"), "success")
        self.assertIn("parse_time_ms", data)
        self.assertGreaterEqual(data["parse_time_ms"], 0.0)
        
        payload = data.get("payload")
        self.assertIsNotNone(payload)
        self.assertTrue(payload["is_valid_spell"])
        self.assertGreaterEqual(payload["ring_circularity"], 0.72)
        self.assertEqual(type(payload["detected_symbols"]), list)

    def test_parse_spell_invalid_base64_syntax(self):
        """
        Verifies that a malformed or corrupted Base64 byte-stream syntax
        properly triggers an HTTP 400 Bad Request exception.
        """
        # Sending non-base64 characters that cause decoding exceptions
        request_body = {
            "session_id": "test_invalid_base64",
            "image_base64": "!!!###$$$invalid_base64_payload$$$###!!!"
        }
        
        response = self.client.post("/api/parse-spell", json=request_body)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertTrue(data["detail"].startswith("Invalid Base64 byte-stream syntax"))

    def test_parse_spell_decoded_matrix_empty(self):
        """
        Verifies that a valid Base64 string that does NOT decode to a valid image matrix
        (e.g., standard text string encoded in Base64) fails gracefully with HTTP 400
        and a specific structural detail payload.
        """
        # "abcdefg" encoded in base64: YWJjZGVmZw==
        # Decodes to plain ASCII text, causing cv2.imdecode to return None.
        request_body = {
            "session_id": "test_corrupted_stream",
            "image_base64": "data:image/png;base64,YWJjZGVmZw=="
        }
        
        response = self.client.post("/api/parse-spell", json=request_body)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertEqual(
            data["detail"],
            "Invalid Base64 byte-stream syntax: Decoded image matrix is empty"
        )

    def test_cors_headers_injection(self):
        """
        Verifies that the injected CORSMiddleware responds with unrestricted access
        headers (*), allowing any origins to query our Wi-Fi bridge local server.
        """
        # Generate headers for simple OPTIONS CORS request pre-flight checks
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
        
        response = self.client.options("/api/parse-spell", headers=headers)
        
        # Check standard CORS response headers
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "*")
        self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))

if __name__ == "__main__":
    unittest.main()
