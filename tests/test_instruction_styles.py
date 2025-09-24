#!/usr/bin/env python3
"""
Test script to demonstrate different instruction template styles
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from scripts.generate_conversation_data import ConversationGenerator

def test_instruction_styles():
    """Test different instruction styles"""

    print("Instruction Template Examples")
    print("=" * 60)

    styles = ["formal", "casual", "technical", "specific"]

    for style in styles:
        print(f"\n{style.upper()} STYLE:")
        print("-" * 40)

        generator = ConversationGenerator(style=style, seed=42)

        # Generate a few instructions
        for i in range(3):
            instruction = generator.get_instruction()
            print(f"{i+1}. {instruction[:100]}..." if len(instruction) > 100 else f"{i+1}. {instruction}")

        # Also show custom entity type instruction
        custom_instruction = generator.get_instruction(entity_types=["PERSON", "ORG", "LOC"])
        print(f"Custom: {custom_instruction}")

    print("\n" + "=" * 60)
    print("Entity-specific Instructions:")
    print("-" * 40)

    generator = ConversationGenerator(style="balanced", seed=42)

    # Different entity combinations
    entity_combinations = [
        ["PERSON", "ORG"],
        ["LOC", "DATE", "EVENT"],
        ["MONEY", "PERCENT", "ORG"],
        ["PERSON", "ORG", "LOC", "DATE", "MONEY"]
    ]

    for entities in entity_combinations:
        instruction = generator.get_instruction(entity_types=entities)
        print(f"\nEntities: {', '.join(entities)}")
        print(f"Instruction: {instruction}")

    print("\n" + "=" * 60)
    print("Complete Example with New Templates:")
    print("-" * 40)

    # Generate a complete example
    generator = ConversationGenerator(style="balanced", seed=42)
    text, entities = generator.generate_simple_example()
    instruction = generator.get_instruction(entity_types=list(entities.keys()))

    print(f"\nText: {text}")
    print(f"\nInstruction: {instruction}")
    print(f"\nExtracted Entities: {json.dumps(entities, indent=2)}")

    print("\n✓ All instruction template styles demonstrated!")

if __name__ == "__main__":
    test_instruction_styles()