#!/usr/bin/env python3
"""
Test script to verify conversation format implementation
"""
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_processor import NERDataProcessor

def test_conversation_format():
    """Test that conversation format is working correctly"""

    print("Testing Conversation Format Implementation\n")
    print("=" * 50)

    # Test 1: Create processor with conversation format
    print("\n1. Testing conversation format data processor...")
    processor = NERDataProcessor(use_conversation_format=True)

    test_text = "Apple Inc. was founded by Steve Jobs in Cupertino."
    test_entities = {
        "ORG": ["Apple Inc."],
        "PERSON": ["Steve Jobs"],
        "GPE": ["Cupertino"]
    }

    formatted = processor.format_for_training(test_text, test_entities)
    print(f"Input text: {test_text}")
    print(f"Entities: {test_entities}")
    print(f"\nFormatted output:")
    print(json.dumps(formatted, indent=2))

    # Verify format
    assert 'messages' in formatted, "Conversation format should have 'messages' field"
    assert len(formatted['messages']) == 2, "Should have user and assistant messages"
    assert formatted['messages'][0]['role'] == 'user', "First message should be from user"
    assert formatted['messages'][1]['role'] == 'assistant', "Second message should be from assistant"
    print("✓ Conversation format test passed")

    # Test 2: Test legacy format
    print("\n2. Testing legacy format data processor...")
    legacy_processor = NERDataProcessor(use_conversation_format=False)

    legacy_formatted = legacy_processor.format_for_training(test_text, test_entities)
    print(f"\nLegacy formatted output:")
    print(json.dumps(legacy_formatted, indent=2))

    assert 'text' in legacy_formatted, "Legacy format should have 'text' field"
    assert 'instruction' in legacy_formatted, "Legacy format should have 'instruction' field"
    assert 'messages' not in legacy_formatted, "Legacy format should not have 'messages' field"
    print("✓ Legacy format test passed")

    # Test 3: Test prompt-completion format
    print("\n3. Testing prompt-completion conversation format...")
    pc_formatted = processor.format_for_prompt_completion(test_text, test_entities)
    print(f"\nPrompt-Completion formatted output:")
    print(json.dumps(pc_formatted, indent=2))

    assert 'prompt' in pc_formatted, "Should have 'prompt' field"
    assert 'completion' in pc_formatted, "Should have 'completion' field"
    assert isinstance(pc_formatted['prompt'], list), "Prompt should be a list for conversation format"
    assert isinstance(pc_formatted['completion'], list), "Completion should be a list for conversation format"
    print("✓ Prompt-completion format test passed")

    # Test 4: Test batch processing
    print("\n4. Testing batch processing for conversation format...")
    batch_examples = {
        'input': [test_text, "Microsoft is based in Seattle."],
        'output': [
            json.dumps(test_entities),
            json.dumps({"ORG": ["Microsoft"], "GPE": ["Seattle"]})
        ],
        'instruction': [processor.instruction_template] * 2
    }

    batch_formatted = processor.prepare_for_sft(batch_examples)
    print(f"\nBatch formatted output (first example):")
    print(json.dumps(batch_formatted['messages'][0], indent=2))

    assert 'messages' in batch_formatted, "Batch should have 'messages' field"
    assert len(batch_formatted['messages']) == 2, "Should have 2 examples"
    print("✓ Batch processing test passed")

    print("\n" + "=" * 50)
    print("All tests passed successfully! ✓")
    print("\nThe conversation format implementation is working correctly.")
    print("You can now use:")
    print("  - Conversation format for instruction-tuned models")
    print("  - Legacy format for backward compatibility")
    print("  - Prompt-completion format for specific use cases")

if __name__ == "__main__":
    test_conversation_format()