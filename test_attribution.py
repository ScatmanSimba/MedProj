import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
from src.features.response_attribution import ResponseAttributor

def test_attribution_quality():
    # Test cases with expected outcomes
    test_cases = [
        {
            "text": "Lexapro made me numb. Wellbutrin gives me energy.",
            "medications": ["Lexapro", "Wellbutrin"],
            "expected": {
                "Lexapro": {"emotional": 0.3},
                "Wellbutrin": {"activation": 0.7}
            }
        },
        {
            "text": "Prozac helps with my anxiety but makes me sleepy.",
            "medications": ["Prozac"],
            "expected": {
                "Prozac": {"activation": 0.3}
            }
        }
    ]
    
    attributor = ResponseAttributor()
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Text: {test_case['text']}")
        print(f"Medications: {test_case['medications']}")
        
        results = attributor.attribute_responses(
            test_case['text'],
            test_case['medications']
        )
        
        print("\nResults:")
        print(json.dumps(results['dimension_scores'], indent=2))
        
        # Verify expected scores
        for med, expected_scores in test_case['expected'].items():
            if med not in results['dimension_scores']:
                print(f"Error: {med} not found in results")
                continue
                
            for dimension, expected_score in expected_scores.items():
                if dimension not in results['dimension_scores'][med]:
                    print(f"Error: {dimension} not found for {med}")
                    continue
                    
                actual_score = results['dimension_scores'][med][dimension]
                if abs(actual_score - expected_score) > 0.2:  # Allow some margin
                    print(f"Warning: {med} {dimension} score {actual_score:.2f} differs from expected {expected_score:.2f}")
                else:
                    print(f"✓ {med} {dimension} score {actual_score:.2f} matches expected {expected_score:.2f}")

if __name__ == "__main__":
    test_attribution_quality() 