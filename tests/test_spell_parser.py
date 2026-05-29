import unittest
import numpy as np
import cv2
import os
import shutil
from spell_parser import DeterministicSpellParser, RingMetrics

class TestDeterministicSpellParser(unittest.TestCase):
    def setUp(self):
        self.parser = DeterministicSpellParser(circularity_threshold=0.72)
        # Clear debug output directory if it exists to ensure fresh test results
        if os.path.exists('./debug_output'):
            try:
                shutil.rmtree('./debug_output')
            except OSError:
                pass

    def test_zhang_suen_thinning(self):
        """
        Verifies that the vectorized Zhang-Suen algorithm reduces a thick stroke
        to a perfect 1-pixel-wide topological centerline skeleton.
        """
        # Create a simple thick horizontal line (100x100 mask)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(mask, (20, 50), (80, 50), 255, 10)  # Stroke thickness is 10px
        
        # Apply skeletonization
        skeleton = self.parser.zhang_suen_thinning(mask)
        
        # Ensure skeleton is generated
        self.assertTrue(np.any(skeleton == 255))
        
        # Confirm centerline width is exactly 1 pixel for the main body of the line
        for x in range(25, 75):
            col_pixels = np.count_nonzero(skeleton[:, x])
            self.assertEqual(col_pixels, 1, f"Skeleton at column {x} is not exactly 1 pixel wide!")

    def test_calculate_median_stroke_width(self):
        """
        Verifies that the distance-transform centerline sampling method accurately
        estimates the median stroke width without any magic numbers.
        """
        # Create a synthetic circle of radius 30 and stroke width 8
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, thickness=8)
        
        # Estimate stroke width
        median_width = self.parser.calculate_median_stroke_width(mask)
        
        # The exact input width is 8. Verify the estimate is highly accurate (within 1.5 pixels delta)
        self.assertAlmostEqual(median_width, 8.0, delta=1.5)

    def test_bridge_gaps(self):
        """
        Verifies that the gap-bridging convolved endpoint pairing algorithm
        successfully heals minor stroke boundary leaks.
        """
        # Create a circle with a small gap (Radius 30, stroke width 4)
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.circle(mask, (60, 60), 30, 255, thickness=4)
        
        # Slice a small gap at the top (x=60, y=30) by drawing a black block
        cv2.rectangle(mask, (58, 25), (62, 35), 0, -1)
        
        # Skeletonize and calculate median width
        skeleton = self.parser.zhang_suen_thinning(mask)
        median_width = self.parser.calculate_median_stroke_width(mask)
        
        # Apply gap bridging
        bridged = self.parser.bridge_gaps(mask, skeleton, median_width)
        
        # Re-skeletonize the bridged mask
        bridged_skeleton = self.parser.zhang_suen_thinning(bridged)
        
        # Locate endpoints on the bridged skeleton using the correct float32 non-saturating convolution
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        skel_float32 = bridged_skeleton.astype(np.float32)
        neighbor_sum = cv2.filter2D(skel_float32, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.argwhere((bridged_skeleton == 255) & (np.isclose(neighbor_sum, 255.0)))
        
        self.assertEqual(len(endpoints), 0, "The gap-bridging engine did not close the circle boundary leak!")

    def test_isolate_parent_ring_closed(self):
        """
        Tests the end-to-end isolation of a single closed circular ring, compiling RingMetrics
        and exporting debug visuals.
        """
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 255, thickness=6)
        
        contour, metrics = self.parser.isolate_parent_ring(mask, debug_inspect=True)
        
        self.assertIsNotNone(contour)
        self.assertIsNotNone(metrics)
        self.assertTrue(metrics.is_closed)
        self.assertGreaterEqual(metrics.circularity_score, 0.72)
        # minEnclosingCircle returns the outer enclosing radius. The outer boundary of radius 60 + thickness/2 (3) is 63.
        # Expect ~63-65px. Use delta=6.0 to comfortably encompass discretization variance.
        self.assertAlmostEqual(metrics.radius_px, 60.0, delta=6.0)
        self.assertAlmostEqual(metrics.centroid[0], 100, delta=2)
        self.assertAlmostEqual(metrics.centroid[1], 100, delta=2)
        
        # Check that debug images were generated successfully
        self.assertTrue(os.path.exists('./debug_output/01_skeleton_endpoints.png'))
        self.assertTrue(os.path.exists('./debug_output/02_isolated_parent_ring.png'))
        self.assertTrue(os.path.exists('./debug_output/03_modifier_rois.png'))

    def test_concentric_loops_resolution(self):
        """
        Tests the concentric double-ring resolution logic: largest area loop must
        be selected as the Topological Root.
        """
        # Outer ring (Radius 80, stroke 6) and Inner ring (Radius 40, stroke 4)
        mask = np.zeros((250, 250), dtype=np.uint8)
        cv2.circle(mask, (125, 125), 80, 255, thickness=6)
        cv2.circle(mask, (125, 125), 40, 255, thickness=4)
        
        contour, metrics = self.parser.isolate_parent_ring(mask, debug_inspect=True)
        
        self.assertIsNotNone(contour)
        self.assertIsNotNone(metrics)
        
        # Verify the outer (Radius 80) is selected as root based on largest area
        self.assertAlmostEqual(metrics.radius_px, 80.0, delta=4.0)

    def test_leaf_symbol_spatial_agglomeration_and_normalization(self):
        """
        Tests Step 3 and Step 4: Spatial Enclosure, Multi-Stroke Agglomeration, 
        Displacement Angle calculation, Localized padded De-rotation, and 28x28 Normalization.
        """
        # Create a synthetic canvas containing:
        # 1. A closed Parent Ring of radius 60 centered at (100, 100)
        # 2. A multi-stroke symbol (e.g. an exclamation-like mark of 2 disconnected lines) 
        #    located off-center at a 45-degree angle (126, 126)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 255, thickness=6)
        
        # Stroke 1: Line of the symbol (oriented outward at 45 degrees)
        cv2.line(mask, (120, 120), (128, 128), 255, thickness=3)
        # Stroke 2: A small dot nearby (distance is about 5 pixels, which is < 1.5 * median_width (~6-9px))
        cv2.circle(mask, (134, 134), 2, 255, -1)
        
        # Run parent ring isolation
        parent_contour, ring_metrics = self.parser.isolate_parent_ring(mask, debug_inspect=True)
        self.assertIsNotNone(parent_contour)
        
        # Run symbol extraction and normalization
        tokens = self.parser.extract_and_normalize_tokens(mask, parent_contour, ring_metrics, debug_inspect=True)
        
        # Verify exactly 1 unified clustered token was extracted (multi-stroke was successfully agglomerated)
        self.assertEqual(len(tokens), 1, "Failed to agglomerate multi-stroke candidate into 1 token!")
        token = tokens[0]
        
        # Verify definitive centroid is in the neighborhood of (126, 126)
        self.assertAlmostEqual(token.centroid[0], 126, delta=3)
        self.assertAlmostEqual(token.centroid[1], 126, delta=3)
        
        # Verify placement angle theta is close to 45 degrees (pi / 4 = 0.785 rad)
        self.assertAlmostEqual(token.placement_angle_rad, np.pi / 4, delta=0.1)
        
        # Verify dual orientation matrix shapes are exactly 28x28 float32 with values [0.0, 1.0]
        self.assertEqual(token.matrix_outward.shape, (28, 28))
        self.assertEqual(token.matrix_inward.shape, (28, 28))
        self.assertEqual(token.matrix_outward.dtype, np.float32)
        self.assertEqual(token.matrix_inward.dtype, np.float32)
        
        # Matrix values should be bounded between 0.0 and 1.0
        self.assertTrue(np.all(token.matrix_outward >= 0.0))
        self.assertTrue(np.all(token.matrix_outward <= 1.0))
        # Ensure it contains some foreground pixels (values > 0)
        self.assertTrue(np.any(token.matrix_outward > 0.0))

        # Verification of the 03_modifier_rois.png output
        self.assertTrue(os.path.exists('./debug_output/03_modifier_rois.png'))

    def test_deterministic_rejection(self):
        """
        Verifies that a random chaotic scribble inside a parent magic ring
        is successfully classified as 'unknown' due to a low confidence score (< 0.70).
        """
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 255, thickness=6)
        
        # Chaotic scribble (criss-cross lines)
        cv2.line(mask, (80, 80), (120, 120), 255, thickness=4)
        cv2.line(mask, (120, 80), (80, 120), 255, thickness=4)
        cv2.circle(mask, (100, 80), 5, 255, -1)
        
        dict_paths = {
            "sigils": "c:\\Users\\ZakiZ\\Documents\\01_projects\\wha-spell-simulator\\src\\dictionary\\sigils.json",
            "signs": "c:\\Users\\ZakiZ\\Documents\\01_projects\\wha-spell-simulator\\src\\dictionary\\signs.json"
        }
        
        final_ir = self.parser.compile_spell_diagram(mask, dict_paths)
        
        self.assertTrue(final_ir.is_valid_spell)
        self.assertEqual(len(final_ir.detected_symbols), 1)
        symbol = final_ir.detected_symbols[0]
        self.assertEqual(symbol.symbol_class, "unknown")
        self.assertLess(symbol.confidence_score, 0.70)

    def test_orientation_disambiguation(self):
        """
        Verifies that orientation (outward vs inward) is resolved via L2 pixel distance (MSE)
        and mapped correctly to the symbol payload.
        """
        dict_paths = {
            "sigils": "c:\\Users\\ZakiZ\\Documents\\01_projects\\wha-spell-simulator\\src\\dictionary\\sigils.json",
            "signs": "c:\\Users\\ZakiZ\\Documents\\01_projects\\wha-spell-simulator\\src\\dictionary\\signs.json"
        }
        
        import json
        with open(dict_paths["signs"], 'r', encoding='utf-8') as f:
            data = json.load(f)
            column_template = next(item for item in data if item["id"] == "column")
            
        # 1. Outward case: Column template pointing OUTWARD (stem pointing up/away from center)
        mask_outward = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask_outward, (100, 100), 60, 255, thickness=6)
        
        # Draw column template completely inside parent ring to avoid contour merge
        # Stem pointing up: vertical line from x=100, y=52 to x=100, y=68
        cv2.line(mask_outward, (100, 52), (100, 68), 255, thickness=3)
        # Base: horizontal line at the bottom y=68 from x=93 to x=107
        cv2.line(mask_outward, (93, 68), (107, 68), 255, thickness=3)
        
        final_ir_outward = self.parser.compile_spell_diagram(mask_outward, dict_paths)
        self.assertEqual(len(final_ir_outward.detected_symbols), 1)
        sym_outward = final_ir_outward.detected_symbols[0]
        self.assertEqual(sym_outward.symbol_class, "column")
        self.assertEqual(sym_outward.orientation, "outward")
        self.assertGreaterEqual(sym_outward.confidence_score, 0.70)
        
        # 2. Inward case: Column template pointing INWARD (stem pointing down/towards center)
        mask_inward = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask_inward, (100, 100), 60, 255, thickness=6)
        
        # Stem pointing down: vertical line from x=100, y=52 to x=100, y=68
        cv2.line(mask_inward, (100, 52), (100, 68), 255, thickness=3)
        # Base: horizontal line at the top y=52 from x=93 to x=107
        cv2.line(mask_inward, (93, 52), (107, 52), 255, thickness=3)
        
        final_ir_inward = self.parser.compile_spell_diagram(mask_inward, dict_paths)
        self.assertEqual(len(final_ir_inward.detected_symbols), 1)
        sym_inward = final_ir_inward.detected_symbols[0]
        self.assertEqual(sym_inward.symbol_class, "column")
        self.assertEqual(sym_inward.orientation, "inward")
        self.assertGreaterEqual(sym_inward.confidence_score, 0.70)

if __name__ == '__main__':
    unittest.main()
