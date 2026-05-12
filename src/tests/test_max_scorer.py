"""
Test MaxScorer to verify it evaluates each position individually and returns max score.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.scorer.max_scorer import MaxScorer
from src.evaluator.base_evaluator import BaseEvaluator
from typing import Dict, Any


class MockEvaluator(BaseEvaluator):
    """Mock evaluator that returns predictable scores based on position."""
    
    def evaluate(self, data: Dict[str, Any]) -> float:
        """Return score based on start position (for testing purposes)."""
        start = data.get('start', 0)
        end = data.get('end', 1)
        
        # Return different scores for different positions
        # Position (2,3) -> 0.8, (3,4) -> 0.9, (4,5) -> 0.7, etc.
        position_scores = {
            (0, 1): 0.5,
            (1, 2): 0.6,
            (2, 3): 0.8,
            (3, 4): 0.9,
            (4, 5): 0.7,
            (5, 6): 0.6,
            (6, 7): 0.4,
        }
        
        return position_scores.get((start, end), 0.5)


def test_max_scorer_range():
    """Test MaxScorer on a range of positions."""
    print("="*60)
    print("TEST 1: MaxScorer on range [2, 6]")
    print("="*60)
    
    evaluator = MockEvaluator()
    scorer = MaxScorer(evaluator)
    
    data = {
        "problem": "Test problem",
        "answer": "Test answer",
        "history": ["step1", "step2", "step3", "step4", "step5", "step6", "step7"],
        "start": 2,
        "end": 6
    }
    
    # Expected: evaluates (2,3)->0.8, (3,4)->0.9, (4,5)->0.7, (5,6)->0.6
    # Should return max = 0.9
    max_score = scorer.score(data)
    
    print(f"Input range: [{data['start']}, {data['end']}]")
    print(f"Expected to evaluate positions: (2,3), (3,4), (4,5), (5,6)")
    print(f"Expected scores: 0.8, 0.9, 0.7, 0.6")
    print(f"Expected max score: 0.9")
    print(f"Actual max score: {max_score}")
    print(f"Test PASSED: {max_score == 0.9}\n")
    
    return max_score == 0.9


def test_max_scorer_single_position():
    """Test MaxScorer on a single position (start == end)."""
    print("="*60)
    print("TEST 2: MaxScorer on single position [3, 3]")
    print("="*60)
    
    evaluator = MockEvaluator()
    scorer = MaxScorer(evaluator)
    
    data = {
        "problem": "Test problem",
        "answer": "Test answer", 
        "history": ["step1", "step2", "step3", "step4"],
        "start": 3,
        "end": 3
    }
    
    # When start == end, it should evaluate that single position
    # But our mock evaluator expects (start, end) pairs, so it will return 0.5 (default)
    score = scorer.score(data)
    
    print(f"Input range: [{data['start']}, {data['end']}]")
    print(f"Single position evaluation")
    print(f"Actual score: {score}")
    print(f"Test PASSED: {score == 0.5}\n")
    
    return score == 0.5


def test_max_scorer_small_range():
    """Test MaxScorer on a small range."""
    print("="*60)
    print("TEST 3: MaxScorer on range [0, 2]")
    print("="*60)
    
    evaluator = MockEvaluator()
    scorer = MaxScorer(evaluator)
    
    data = {
        "problem": "Test problem",
        "answer": "Test answer",
        "history": ["step1", "step2", "step3"],
        "start": 0,
        "end": 2
    }
    
    # Expected: evaluates (0,1)->0.5, (1,2)->0.6
    # Should return max = 0.6
    max_score = scorer.score(data)
    
    print(f"Input range: [{data['start']}, {data['end']}]")
    print(f"Expected to evaluate positions: (0,1), (1,2)")
    print(f"Expected scores: 0.5, 0.6")
    print(f"Expected max score: 0.6")
    print(f"Actual max score: {max_score}")
    print(f"Test PASSED: {max_score == 0.6}\n")
    
    return max_score == 0.6


def run_all_tests():
    """Run all MaxScorer tests."""
    print("\n" + "="*60)
    print("RUNNING MAX SCORER TESTS")
    print("="*60 + "\n")
    
    results = []
    results.append(("Range [2, 6]", test_max_scorer_range()))
    results.append(("Single position [3, 3]", test_max_scorer_single_position()))
    results.append(("Small range [0, 2]", test_max_scorer_small_range()))
    
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed!'}")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
