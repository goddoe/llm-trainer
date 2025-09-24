#!/usr/bin/env python3
"""
Demonstration of structured output vs regular NER
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scripts.generate_structured_data import StructuredOutputGenerator
from scripts.generate_conversation_data import ConversationGenerator


def demo_comparison():
    print("=" * 70)
    print("STRUCTURED OUTPUT vs REGULAR NER COMPARISON")
    print("=" * 70)

    # Same text for both
    test_text = "Sarah Johnson, the CEO at Microsoft, announced a high priority technology project scheduled for Q1 2024 in San Francisco."

    print("\nInput Text:")
    print("-" * 40)
    print(test_text)

    # Regular NER
    print("\n" + "=" * 70)
    print("1. REGULAR NER (Extract only)")
    print("-" * 40)

    regular_instruction = "Extract entities for: PERSON, ORG, LOC, DATE. Return the results in JSON format."
    regular_output = {
        "PERSON": ["Sarah Johnson"],
        "ORG": ["Microsoft"],
        "LOC": ["San Francisco"],
        "DATE": ["Q1 2024"]
    }

    print(f"Instruction: {regular_instruction}")
    print(f"\nOutput:")
    print(json.dumps(regular_output, indent=2))
    print("\n✓ Extracts entities found in text")
    print("✗ No additional structured information")

    # Structured Output
    print("\n" + "=" * 70)
    print("2. STRUCTURED OUTPUT (Extract + Constrained Choices)")
    print("-" * 40)

    structured_instruction = """Extract structured data according to this schema:
- PERSON: extract person names from text
- ORG: extract organization names from text
- LOC: extract locations from text
- DATE: extract dates from text
- TITLE: select from [CEO, CTO, CFO, VP, Director, Manager]
- PRIORITY: select from [high, medium, low]
- CATEGORY: select from [technology, finance, healthcare, education]
- STATUS: select from [pending, in_progress, completed]
- IS_URGENT: boolean (true/false)

Return as JSON with exact field names."""

    structured_output = {
        "PERSON": ["Sarah Johnson"],
        "ORG": ["Microsoft"],
        "LOC": ["San Francisco"],
        "DATE": ["Q1 2024"],
        "TITLE": "CEO",
        "PRIORITY": "high",
        "CATEGORY": "technology",
        "STATUS": "pending",
        "IS_URGENT": True
    }

    print(f"Instruction:\n{structured_instruction}")
    print(f"\nOutput:")
    print(json.dumps(structured_output, indent=2))
    print("\n✓ Extracts entities from text")
    print("✓ Adds structured fields with constrained values")
    print("✓ Enforces schema compliance")

    # Key Differences
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("-" * 40)

    differences = [
        ("Extraction", "Both extract entities from text", "✓", "✓"),
        ("Constrained Fields", "Predefined choice selection", "✗", "✓"),
        ("Schema Validation", "Enforces output structure", "✗", "✓"),
        ("Boolean Fields", "True/False decisions", "✗", "✓"),
        ("Enum Fields", "Select from valid options", "✗", "✓"),
        ("Type Safety", "Ensures correct data types", "✗", "✓"),
    ]

    print(f"{'Feature':<25} {'Description':<30} {'NER':<5} {'Structured'}")
    print("-" * 70)
    for feature, desc, ner, structured in differences:
        print(f"{feature:<25} {desc:<30} {ner:<5} {structured}")

    # Use Cases
    print("\n" + "=" * 70)
    print("USE CASES")
    print("-" * 40)

    print("\nRegular NER is good for:")
    print("• Information extraction from documents")
    print("• Entity recognition in unstructured text")
    print("• Simple tagging tasks")

    print("\nStructured Output is ideal for:")
    print("• Form filling from text")
    print("• Database population with validation")
    print("• API responses with strict schemas")
    print("• Business rule compliance")
    print("• Workflow automation with typed data")

    # Training Benefits
    print("\n" + "=" * 70)
    print("TRAINING BENEFITS OF STRUCTURED OUTPUT")
    print("-" * 40)

    print("1. **Schema Compliance**: Models learn to follow exact output schemas")
    print("2. **Value Constraints**: Models learn valid value ranges and options")
    print("3. **Type Safety**: Output guarantees correct data types")
    print("4. **Business Logic**: Can encode business rules in constraints")
    print("5. **Reduced Errors**: Eliminates invalid values in production")

    print("\n" + "=" * 70)


def generate_examples():
    """Generate some example structured outputs"""
    print("\n" + "=" * 70)
    print("GENERATED EXAMPLES")
    print("=" * 70)

    generator = StructuredOutputGenerator(seed=42)

    for i in range(3):
        print(f"\nExample {i+1}:")
        print("-" * 40)

        text, output = generator.generate_example()
        fields = list(output.keys())

        # Show which fields are constrained vs extracted
        constrained = []
        extracted = []

        for field in fields:
            if field in generator.schema:
                if generator.schema[field]["type"].value == "extract":
                    extracted.append(field)
                else:
                    constrained.append(field)

        print(f"Text: {text}")
        print(f"\nExtracted fields: {', '.join(extracted)}")
        print(f"Constrained fields: {', '.join(constrained)}")
        print(f"\nStructured Output:")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    demo_comparison()
    generate_examples()
    print("\n✓ Demo completed!")