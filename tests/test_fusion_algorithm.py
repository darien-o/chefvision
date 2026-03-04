"""
Unit tests for the fusion detection algorithm.

Tests the core fusion algorithm including IoU calculation,
detection clustering, weighted voting, and coordinate averaging.
"""

import pytest


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (copied from detect_fusion.py for testing)."""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    union = (x2-x1)*(y2-y1) + (x2b-x1b)*(y2b-y1b) - inter
    return inter / union if union > 0 else 0


def fusion_detection(all_detections, iou_threshold=0.5, min_votes=1):
    """
    Fusion algorithm: Combines detections from multiple models (copied from detect_fusion.py for testing).
    - Groups overlapping boxes from different models
    - Weighted voting for final confidence
    - Averages box coordinates
    """
    from collections import defaultdict
    
    if not all_detections:
        return []
    
    # Group detections by spatial proximity
    clusters = []
    
    for det in all_detections:
        x1, y1, x2, y2, conf, name, model_name, weight = det
        box = (x1, y1, x2, y2)
        
        # Find matching cluster
        matched = False
        for cluster in clusters:
            # Check IoU with cluster representative
            rep_box = cluster['boxes'][0][:4]
            if calculate_iou(box, rep_box) > iou_threshold:
                cluster['boxes'].append(det)
                cluster['votes'] += 1
                cluster['total_weight'] += weight
                matched = True
                break
        
        if not matched:
            clusters.append({
                'boxes': [det],
                'votes': 1,
                'total_weight': weight
            })
    
    # Process clusters into final detections
    final_detections = []
    
    for cluster in clusters:
        if cluster['votes'] < min_votes:
            continue
        
        # Weighted average of boxes and confidence
        total_weight = cluster['total_weight']
        avg_x1 = avg_y1 = avg_x2 = avg_y2 = avg_conf = 0
        class_votes = defaultdict(float)
        
        for det in cluster['boxes']:
            x1, y1, x2, y2, conf, name, model_name, weight = det
            w = weight / total_weight
            
            avg_x1 += x1 * w
            avg_y1 += y1 * w
            avg_x2 += x2 * w
            avg_y2 += y2 * w
            avg_conf += conf * weight
            class_votes[name] += weight
        
        # Select class with highest weighted vote
        best_class = max(class_votes.items(), key=lambda x: x[1])[0]
        
        final_detections.append({
            'box': (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)),
            'confidence': avg_conf / total_weight,
            'class': best_class,
            'votes': cluster['votes'],
            'models': len(cluster['boxes'])
        })
    
    return final_detections


class TestIoUCalculation:
    """Test cases for Intersection over Union calculation."""
    
    def test_identical_boxes(self):
        """IoU of identical boxes should be 1.0."""
        box1 = (10, 10, 50, 50)
        box2 = (10, 10, 50, 50)
        assert calculate_iou(box1, box2) == 1.0
    
    def test_no_overlap(self):
        """IoU of non-overlapping boxes should be 0.0."""
        box1 = (10, 10, 50, 50)
        box2 = (60, 60, 100, 100)
        assert calculate_iou(box1, box2) == 0.0
    
    def test_partial_overlap(self):
        """IoU of partially overlapping boxes should be between 0 and 1."""
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)
        iou = calculate_iou(box1, box2)
        assert 0 < iou < 1
        # Expected IoU: intersection = 20*20 = 400
        # union = 40*40 + 40*40 - 400 = 3200 - 400 = 2800
        # iou = 400/2800 ≈ 0.143
        assert abs(iou - 0.143) < 0.01
    
    def test_iou_symmetry(self):
        """IoU should be symmetric: IoU(A,B) == IoU(B,A)."""
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)
        assert calculate_iou(box1, box2) == calculate_iou(box2, box1)
    
    def test_contained_box(self):
        """IoU when one box is contained in another."""
        box1 = (10, 10, 100, 100)  # Large box
        box2 = (30, 30, 50, 50)    # Small box inside
        iou = calculate_iou(box1, box2)
        # intersection = 20*20 = 400
        # union = 90*90 + 20*20 - 400 = 8100 + 400 - 400 = 8100
        # iou = 400/8100 ≈ 0.049
        assert 0 < iou < 1


class TestFusionDetection:
    """Test cases for the fusion detection algorithm."""
    
    def test_empty_detections(self):
        """Fusion with no detections should return empty list."""
        result = fusion_detection([])
        assert result == []
    
    def test_single_detection(self):
        """Fusion with single detection should return that detection."""
        detections = [
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0)
        ]
        result = fusion_detection(detections, iou_threshold=0.5, min_votes=1)
        
        assert len(result) == 1
        assert result[0]['class'] == "apple"
        assert result[0]['votes'] == 1
        assert result[0]['box'] == (10, 10, 50, 50)
    
    def test_non_overlapping_detections(self):
        """Non-overlapping detections should remain separate."""
        detections = [
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
            (100, 100, 150, 150, 0.8, "banana", "model2", 1.0)
        ]
        result = fusion_detection(detections, iou_threshold=0.5, min_votes=1)
        
        assert len(result) == 2
        classes = {det['class'] for det in result}
        assert classes == {"apple", "banana"}
    
    def test_overlapping_same_class(self):
        """Overlapping detections of same class should be fused."""
        detections = [
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
            (15, 15, 55, 55, 0.85, "apple", "model2", 1.0)
        ]
        result = fusion_detection(detections, iou_threshold=0.5, min_votes=1)
        
        assert len(result) == 1
        assert result[0]['class'] == "apple"
        assert result[0]['votes'] == 2
        # Coordinates should be averaged
        assert result[0]['box'] == (12, 12, 52, 52)
    
    def test_weighted_voting(self):
        """Higher weighted model should influence class selection."""
        detections = [
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
            (15, 15, 55, 55, 0.85, "banana", "model2", 2.0)  # Higher weight
        ]
        result = fusion_detection(detections, iou_threshold=0.5, min_votes=1)
        
        assert len(result) == 1
        # Banana should win due to higher weight
        assert result[0]['class'] == "banana"
        assert result[0]['votes'] == 2
    
    def test_min_votes_filtering(self):
        """Detections below min_votes threshold should be filtered."""
        detections = [
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
            (100, 100, 150, 150, 0.8, "banana", "model2", 1.0)
        ]
        result = fusion_detection(detections, iou_threshold=0.5, min_votes=2)
        
        # Both detections have only 1 vote, should be filtered
        assert len(result) == 0
    
    def test_confidence_averaging(self):
        """Confidence should be weighted average."""
        detections = [
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
            (15, 15, 55, 55, 0.6, "apple", "model2", 1.0)
        ]
        result = fusion_detection(detections, iou_threshold=0.5, min_votes=1)
        
        assert len(result) == 1
        # Average confidence: (0.9 * 1.0 + 0.6 * 1.0) / 2.0 = 0.75
        assert abs(result[0]['confidence'] - 0.75) < 0.01
    
    def test_multiple_clusters(self):
        """Multiple separate clusters should be detected correctly."""
        detections = [
            # Cluster 1: apple
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
            (15, 15, 55, 55, 0.85, "apple", "model2", 1.0),
            # Cluster 2: banana
            (100, 100, 150, 150, 0.8, "banana", "model1", 1.0),
            (105, 105, 155, 155, 0.75, "banana", "model2", 1.0),
        ]
        result = fusion_detection(detections, iou_threshold=0.5, min_votes=1)
        
        assert len(result) == 2
        classes = {det['class'] for det in result}
        assert classes == {"apple", "banana"}
        
        # Each cluster should have 2 votes
        for det in result:
            assert det['votes'] == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_area_box(self):
        """Handle boxes with zero area gracefully."""
        box1 = (10, 10, 10, 10)  # Zero area
        box2 = (10, 10, 50, 50)
        iou = calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_high_iou_threshold(self):
        """Very high IoU threshold should prevent clustering."""
        detections = [
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
            (15, 15, 55, 55, 0.85, "apple", "model2", 1.0)
        ]
        result = fusion_detection(detections, iou_threshold=0.99, min_votes=1)
        
        # Should not cluster due to high threshold
        assert len(result) == 2
    
    def test_low_iou_threshold(self):
        """Very low IoU threshold should cluster more aggressively."""
        detections = [
            (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
            (40, 40, 80, 80, 0.85, "apple", "model2", 1.0)
        ]
        result = fusion_detection(detections, iou_threshold=0.01, min_votes=1)
        
        # Should cluster even with small overlap
        assert len(result) == 1


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
