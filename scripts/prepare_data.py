#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import NERDataProcessor


def convert_conll_to_jsonl(output_dir: str, max_samples: int = None, use_conversation_format: bool = True):
    processor = NERDataProcessor(use_conversation_format=use_conversation_format)
    dataset = load_dataset("conll2003")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        data_split = dataset[split_name]
        if max_samples:
            data_split = data_split.select(range(min(max_samples, len(data_split))))

        examples = []
        for item in data_split:
            tokens = item['tokens']
            ner_tags = item['ner_tags']

            text = ' '.join(tokens)
            entities = processor._extract_entities_from_tags(
                tokens, ner_tags, "conll2003"
            )

            formatted = processor.format_for_training(text, entities)
            examples.append(formatted)

        output_file = output_path / f"{split_name}.jsonl"
        processor.save_jsonl(examples, output_file)
        print(f"Saved {len(examples)} examples to {output_file}")


def convert_custom_csv_to_jsonl(
    csv_path: str,
    output_dir: str,
    text_column: str = "text",
    entities_column: str = "entities",
    test_size: float = 0.1,
    val_size: float = 0.1,
    use_conversation_format: bool = True,
):
    processor = NERDataProcessor(use_conversation_format=use_conversation_format)

    df = pd.read_csv(csv_path)

    if entities_column in df.columns:
        # Assume entities are stored as JSON strings
        df['entities_parsed'] = df[entities_column].apply(
            lambda x: json.loads(x) if pd.notna(x) else {}
        )
    else:
        raise ValueError(f"Column {entities_column} not found in CSV")

    # Prepare data
    examples = []
    for _, row in df.iterrows():
        text = row[text_column]
        entities = row['entities_parsed']

        # Use varied instruction if in conversation format
        if use_conversation_format and instructions:
            # Override processor's default instruction
            instruction = random.choice(instructions)
            processor.instruction_template = instruction

        formatted = processor.format_for_training(text, entities)
        examples.append(formatted)

    # Split data
    train_data, test_data = train_test_split(
        examples, test_size=test_size, random_state=42
    )
    train_data, val_data = train_test_split(
        train_data, test_size=val_size/(1-test_size), random_state=42
    )

    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processor.save_jsonl(train_data, output_path / "train.jsonl")
    processor.save_jsonl(val_data, output_path / "validation.jsonl")
    processor.save_jsonl(test_data, output_path / "test.jsonl")

    print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


def convert_ontonotes_to_jsonl(output_dir: str, max_samples: int = None, use_conversation_format: bool = True):
    processor = NERDataProcessor(use_conversation_format=use_conversation_format)

    # Note: OntoNotes requires manual download or specific access
    # This is a placeholder for OntoNotes conversion
    print("OntoNotes conversion requires manual setup.")
    print("Please download OntoNotes 5.0 and use appropriate tools.")

    # Example structure for OntoNotes-style data
    example_data = [
        {
            "instruction": "Extract named entities from the following text and return them in JSON format.",
            "input": "Apple Inc. was founded by Steve Jobs in Cupertino.",
            "output": json.dumps({
                "ORG": ["Apple Inc."],
                "PERSON": ["Steve Jobs"],
                "GPE": ["Cupertino"]
            }, indent=2)
        }
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processor.save_jsonl(example_data, output_path / "example.jsonl")
    print(f"Saved example data to {output_path / 'example.jsonl'}")


def create_synthetic_ner_data(output_dir: str, num_samples: int = 1000, use_conversation_format: bool = True, instruction_style: str = "balanced"):
    processor = NERDataProcessor(use_conversation_format=use_conversation_format)

    # Instruction variations for conversation format with entity keys
    instruction_templates = {
        "formal": [
            "Extract entities for: PERSON, ORG, LOC, DATE, MONEY. Return the results in JSON format.",
            "Identify the following entity types: PERSON (names), ORG (organizations), LOC (locations). Format output as JSON.",
            "Perform entity extraction for keys: PERSON, ORG, LOC, DATE, EVENT. Return JSON with entity types as keys.",
        ],
        "casual": [
            "Find: PERSON, ORG, LOC entities. Return as JSON.",
            "Extract people (PERSON), companies (ORG), and places (LOC). Give me JSON format.",
            "Get me: PERSON, ORG, LOC, DATE entities. Format as JSON.",
        ],
        "technical": [
            "Execute NER for types: {PERSON: [], ORG: [], LOC: [], DATE: []}. Return populated JSON.",
            "Apply extraction for entity schema: PERSON|ORG|LOC|DATE|MONEY. Output as JSON object.",
        ],
        "balanced": []  # Will mix all styles
    }

    # Mix all styles for balanced
    if instruction_style == "balanced":
        for style_templates in instruction_templates.values():
            if style_templates:  # Skip empty balanced list
                instruction_templates["balanced"].extend(style_templates)

    instructions = instruction_templates.get(instruction_style, instruction_templates["formal"])

    # Create synthetic NER data for testing
    templates = [
        ("The meeting with {person} from {org} is scheduled in {location}.",
         {"PERSON": ["{person}"], "ORG": ["{org}"], "LOC": ["{location}"]}),
        ("{person} works as a {title} at {org} in {location}.",
         {"PERSON": ["{person}"], "TITLE": ["{title}"], "ORG": ["{org}"], "LOC": ["{location}"]}),
        ("The {event} will take place on {date} at {location}.",
         {"EVENT": ["{event}"], "DATE": ["{date}"], "LOC": ["{location}"]}),
        ("{org} announced revenues of {money} in {date}.",
         {"ORG": ["{org}"], "MONEY": ["{money}"], "DATE": ["{date}"]}),
    ]

    persons = ["John Smith", "Emma Wilson", "David Lee", "Sarah Johnson", "Michael Brown"]
    orgs = ["Google", "Microsoft", "Apple", "Amazon", "Meta"]
    locations = ["New York", "San Francisco", "London", "Tokyo", "Berlin"]
    titles = ["Software Engineer", "Product Manager", "Data Scientist", "CEO", "CTO"]
    events = ["Annual Conference", "Product Launch", "Board Meeting", "Tech Summit"]
    dates = ["January 2024", "March 15th", "next Monday", "Q3 2024"]
    money = ["$1 billion", "$500 million", "$2.5 billion", "$750 million"]

    import random
    random.seed(42)

    # Choose instructions for conversation format
    if use_conversation_format:
        instructions = instruction_templates.get(instruction_style, instruction_templates["formal"])
    else:
        instructions = []

    examples = []
    for _ in range(num_samples):
        template_text, template_entities = random.choice(templates)

        # Fill in the template
        replacements = {
            "person": random.choice(persons),
            "org": random.choice(orgs),
            "location": random.choice(locations),
            "title": random.choice(titles),
            "event": random.choice(events),
            "date": random.choice(dates),
            "money": random.choice(money),
        }

        text = template_text
        entities = {}

        for placeholder, value in replacements.items():
            if f"{{{placeholder}}}" in template_text:
                text = text.replace(f"{{{placeholder}}}", value)

        for entity_type, entity_list in template_entities.items():
            entities[entity_type] = []
            for entity_template in entity_list:
                for placeholder, value in replacements.items():
                    if f"{{{placeholder}}}" in entity_template:
                        entity_value = entity_template.replace(f"{{{placeholder}}}", value)
                        entities[entity_type].append(entity_value)

        # Use varied instruction if in conversation format
        if use_conversation_format and instructions:
            # Override processor's default instruction
            instruction = random.choice(instructions)
            processor.instruction_template = instruction

        formatted = processor.format_for_training(text, entities)
        examples.append(formatted)

    # Split data
    train_size = int(0.8 * len(examples))
    val_size = int(0.1 * len(examples))

    train_data = examples[:train_size]
    val_data = examples[train_size:train_size + val_size]
    test_data = examples[train_size + val_size:]

    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processor.save_jsonl(train_data, output_path / "train.jsonl")
    processor.save_jsonl(val_data, output_path / "validation.jsonl")
    processor.save_jsonl(test_data, output_path / "test.jsonl")

    print(f"Generated synthetic data - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare NER data for training in conversation format")
    parser.add_argument("--dataset", type=str, default="conll2003",
                       choices=["conll2003", "ontonotes", "custom", "synthetic"],
                       help="Dataset to convert")
    parser.add_argument("--input", type=str, help="Input file path (for custom dataset)")
    parser.add_argument("--output", type=str, default="./data",
                       help="Output directory for JSONL files")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum number of samples to process")
    parser.add_argument("--text-column", type=str, default="text",
                       help="Name of text column in CSV (for custom dataset)")
    parser.add_argument("--entities-column", type=str, default="entities",
                       help="Name of entities column in CSV (for custom dataset)")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--conversation-format", action="store_true", default=True,
                       help="Use conversation format for training data (default)")
    parser.add_argument("--legacy-format", action="store_true",
                       help="Use legacy text format instead of conversation format")
    parser.add_argument("--instruction-style", type=str, default="balanced",
                       choices=["formal", "casual", "technical", "balanced"],
                       help="Style of instructions in conversation format")

    args = parser.parse_args()

    # Determine format
    use_conversation_format = not args.legacy_format

    if args.dataset == "conll2003":
        convert_conll_to_jsonl(args.output, args.max_samples, use_conversation_format)
    elif args.dataset == "ontonotes":
        convert_ontonotes_to_jsonl(args.output, args.max_samples, use_conversation_format)
    elif args.dataset == "custom":
        if not args.input:
            raise ValueError("--input required for custom dataset")
        convert_custom_csv_to_jsonl(
            args.input,
            args.output,
            args.text_column,
            args.entities_column,
            use_conversation_format=use_conversation_format
        )
    elif args.dataset == "synthetic":
        create_synthetic_ner_data(args.output, args.num_samples, use_conversation_format, args.instruction_style)

    print(f"\nData preparation completed!")
    print(f"Format: {'Conversation' if use_conversation_format else 'Legacy'}")
    if use_conversation_format:
        print(f"Instruction style: {args.instruction_style}")
    print(f"Files saved to: {args.output}")
    print("\n✓ Ready for training with instruction-tuned models!" if use_conversation_format else "")


if __name__ == "__main__":
    main()