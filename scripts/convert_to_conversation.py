#!/usr/bin/env python3
"""
Convert legacy format NER data to conversation format
Useful for migrating existing datasets to the new format
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent))
from src.data_processor import NERDataProcessor


def convert_legacy_to_conversation(legacy_data: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Convert a single legacy format entry to conversation format"""

    # Handle different legacy formats
    if "instruction" in legacy_data and "input" in legacy_data and "output" in legacy_data:
        # Standard legacy format
        instruction = legacy_data["instruction"]
        input_text = legacy_data["input"]
        output = legacy_data["output"]

        user_content = f"{instruction}\n\nText: {input_text}" if instruction else f"Text: {input_text}"
    elif "text" in legacy_data:
        # Text-only format - try to parse it
        text = legacy_data["text"]
        if "Entities:" in text:
            parts = text.split("Entities:")
            user_content = parts[0].strip()
            output = parts[1].strip() if len(parts) > 1 else "{}"
        else:
            # Can't parse, use as-is
            user_content = text
            output = "{}"
    elif "prompt" in legacy_data and "completion" in legacy_data:
        # Prompt-completion format
        user_content = legacy_data["prompt"]
        output = legacy_data["completion"]
    else:
        raise ValueError(f"Unknown legacy format: {legacy_data.keys()}")

    # Create conversation format
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]

    return {"messages": messages}


def convert_file(input_file: Path, output_file: Path,
                 add_variations: bool = False,
                 instruction_style: str = "formal") -> int:
    """Convert a JSONL file from legacy to conversation format"""

    processor = NERDataProcessor(use_conversation_format=True)
    converted_data = []
    skipped = 0

    # Instruction variations for diversity with entity keys
    instruction_variations = {
        "formal": [
            "Extract entities for: PERSON, ORG, LOC, DATE, MONEY. Return the results in JSON format.",
            "Identify the following entity types: PERSON (names), ORG (organizations), LOC (locations), DATE (dates). Format output as JSON.",
            "Perform extraction for keys: PERSON, ORG, LOC, EVENT, PRODUCT. Return JSON with entity types as keys and lists as values.",
        ],
        "casual": [
            "Find: PERSON, ORG, LOC entities. Return as JSON.",
            "Extract people (PERSON), companies (ORG), places (LOC), dates (DATE). Give me JSON format.",
            "Get me: PERSON, ORG, LOC, MONEY entities. Format as JSON please.",
        ],
        "technical": [
            "Execute NER for schema: {PERSON: [], ORG: [], LOC: [], DATE: [], MONEY: []}. Return populated JSON.",
            "Apply extraction for types: PERSON|ORG|LOC|DATE|MONEY|EVENT. Output as JSON object.",
            "Process with entity keys: [PERSON, ORG, LOC, DATE]. Return JSON formatted results.",
        ]
    }

    instructions = instruction_variations.get(instruction_style, instruction_variations["formal"])
    instruction_idx = 0

    # Read and convert each line
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                legacy_data = json.loads(line)

                # Convert to conversation format
                converted = convert_legacy_to_conversation(legacy_data)

                # Optionally add instruction variations
                if add_variations and "Extract named entities" in converted["messages"][0]["content"]:
                    # Replace with varied instruction
                    new_instruction = instructions[instruction_idx % len(instructions)]
                    instruction_idx += 1

                    # Replace the instruction part
                    content = converted["messages"][0]["content"]
                    if "\n\nText:" in content:
                        text_part = content.split("\n\nText:")[1]
                        converted["messages"][0]["content"] = f"{new_instruction}\n\nText:{text_part}"

                converted_data.append(converted)

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Skipping line {line_num} due to error: {e}")
                skipped += 1
                continue

    # Save converted data
    processor.save_jsonl(converted_data, output_file)

    return len(converted_data), skipped


def main():
    parser = argparse.ArgumentParser(description="Convert legacy NER data to conversation format")
    parser.add_argument("input", type=str, help="Input JSONL file in legacy format")
    parser.add_argument("--output", type=str, help="Output JSONL file (default: input_conversation.jsonl)")
    parser.add_argument("--add-variations", action="store_true",
                       help="Add instruction variations for diversity")
    parser.add_argument("--instruction-style", type=str, default="formal",
                       choices=["formal", "casual", "technical"],
                       help="Style of instruction variations")
    parser.add_argument("--batch", action="store_true",
                       help="Process all .jsonl files in the input directory")

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.batch and input_path.is_dir():
        # Batch processing
        print(f"Processing all JSONL files in {input_path}")
        total_converted = 0
        total_skipped = 0

        for jsonl_file in input_path.glob("*.jsonl"):
            if "conversation" in jsonl_file.name:
                print(f"Skipping {jsonl_file.name} (already converted)")
                continue

            output_file = jsonl_file.parent / f"{jsonl_file.stem}_conversation.jsonl"

            print(f"\nConverting {jsonl_file.name}...")
            converted, skipped = convert_file(
                jsonl_file,
                output_file,
                add_variations=args.add_variations,
                instruction_style=args.instruction_style
            )

            total_converted += converted
            total_skipped += skipped
            print(f"✓ Converted {converted} examples, skipped {skipped}")
            print(f"  Saved to {output_file}")

        print(f"\n" + "="*50)
        print(f"Batch conversion completed!")
        print(f"Total converted: {total_converted}")
        print(f"Total skipped: {total_skipped}")

    else:
        # Single file processing
        if not input_path.exists():
            print(f"Error: Input file {input_path} not found")
            sys.exit(1)

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_conversation.jsonl"

        print(f"Converting {input_path} to conversation format...")
        print(f"Output: {output_path}")

        if args.add_variations:
            print(f"Adding instruction variations ({args.instruction_style} style)")

        converted, skipped = convert_file(
            input_path,
            output_path,
            add_variations=args.add_variations,
            instruction_style=args.instruction_style
        )

        print(f"\n✓ Conversion completed!")
        print(f"Converted: {converted} examples")
        if skipped > 0:
            print(f"Skipped: {skipped} examples (due to errors)")

        # Show sample
        print("\n" + "="*50)
        print("Sample converted conversation:")
        print("="*50)

        processor = NERDataProcessor(use_conversation_format=True)
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                sample = json.loads(first_line)
                print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()