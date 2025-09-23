#!/usr/bin/env python3
"""
Structured output data generator for NER training
Generates training data with both constrained choices and open-ended extraction
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import sys
from enum import Enum

sys.path.append(str(Path(__file__).parent.parent))
from src.data_processor import NERDataProcessor


class FieldType(Enum):
    """Types of fields in structured output"""
    EXTRACT = "extract"  # Extract from text
    ENUM = "enum"  # Select from predefined choices
    BOOLEAN = "boolean"  # True/False
    NUMBER = "number"  # Numeric value (with optional range)


class StructuredOutputGenerator:
    """Generate training data for structured output with constraints"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

        # Define schema with constrained and open fields
        self.schema = {
            # Open-ended extraction fields
            "PERSON": {
                "type": FieldType.EXTRACT,
                "description": "person names found in text",
                "multiple": True
            },
            "ORG": {
                "type": FieldType.EXTRACT,
                "description": "organization/company names",
                "multiple": True
            },
            "LOC": {
                "type": FieldType.EXTRACT,
                "description": "geographical locations",
                "multiple": True
            },
            "DATE": {
                "type": FieldType.EXTRACT,
                "description": "dates and time expressions",
                "multiple": True
            },
            "MONEY": {
                "type": FieldType.EXTRACT,
                "description": "monetary amounts",
                "multiple": True
            },

            # Constrained choice fields
            "TITLE": {
                "type": FieldType.ENUM,
                "choices": ["CEO", "CTO", "CFO", "COO", "VP", "Director", "Manager", "Engineer", "Analyst", "Consultant"],
                "description": "job title",
                "multiple": False
            },
            "DEPARTMENT": {
                "type": FieldType.ENUM,
                "choices": ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Legal", "R&D"],
                "description": "department",
                "multiple": False
            },
            "CATEGORY": {
                "type": FieldType.ENUM,
                "choices": ["technology", "finance", "healthcare", "education", "retail", "manufacturing", "government"],
                "description": "industry category",
                "multiple": False
            },
            "SENTIMENT": {
                "type": FieldType.ENUM,
                "choices": ["positive", "negative", "neutral"],
                "description": "sentiment",
                "multiple": False
            },
            "PRIORITY": {
                "type": FieldType.ENUM,
                "choices": ["high", "medium", "low"],
                "description": "priority level",
                "multiple": False
            },
            "STATUS": {
                "type": FieldType.ENUM,
                "choices": ["pending", "in_progress", "completed", "cancelled"],
                "description": "status",
                "multiple": False
            },

            # Boolean fields
            "IS_URGENT": {
                "type": FieldType.BOOLEAN,
                "description": "urgency flag",
                "multiple": False
            },
            "REQUIRES_FOLLOWUP": {
                "type": FieldType.BOOLEAN,
                "description": "requires follow-up",
                "multiple": False
            }
        }

        # Data pools for generation
        self.data_pools = {
            "persons": [
                "John Smith", "Emma Wilson", "David Chen", "Maria Garcia", "Ahmed Hassan",
                "Sophie Martin", "James Anderson", "Liu Wei", "Anna Kowalski", "Roberto Silva"
            ],
            "organizations": [
                "Google", "Microsoft", "OpenAI", "Meta", "Apple", "Amazon", "Tesla",
                "World Bank", "United Nations", "Harvard University", "MIT", "Stanford"
            ],
            "locations": [
                "New York", "San Francisco", "London", "Tokyo", "Paris", "Berlin",
                "Singapore", "Dubai", "Sydney", "Toronto", "Mumbai", "Beijing"
            ],
            "dates": [
                "January 2024", "March 15th", "next Monday", "Q3 2024", "last year",
                "2023", "December 31st", "this week", "yesterday", "tomorrow"
            ],
            "money": [
                "$1 billion", "$500 million", "€250,000", "¥1,000,000", "$50K",
                "£2.5 million", "$750M", "€1.2B", "$15,000", "$3.7 trillion"
            ]
        }

        # Templates for generating text with structured data
        self.templates = [
            {
                "text": "{person}, {title} at {org}, announced a {priority} priority project in {category} sector scheduled for {date}.",
                "fields": ["PERSON", "TITLE", "ORG", "PRIORITY", "CATEGORY", "DATE"],
                "values": {
                    "IS_URGENT": lambda p: p == "high",
                    "STATUS": "pending"
                }
            },
            {
                "text": "The {department} department at {org} led by {person} ({title}) reported {sentiment} results with {money} revenue.",
                "fields": ["DEPARTMENT", "ORG", "PERSON", "TITLE", "SENTIMENT", "MONEY"],
                "values": {
                    "CATEGORY": lambda: random.choice(["technology", "finance"]),
                    "REQUIRES_FOLLOWUP": lambda s: s == "negative"
                }
            },
            {
                "text": "{org} in {location} is hiring a {title} for their {department} team, starting {date}.",
                "fields": ["ORG", "LOC", "TITLE", "DEPARTMENT", "DATE"],
                "values": {
                    "STATUS": "in_progress",
                    "CATEGORY": lambda: random.choice(["technology", "healthcare"])
                }
            },
            {
                "text": "Meeting with {person} and {person2} from {org} about the {priority} priority {category} initiative is scheduled for {date} in {location}.",
                "fields": ["PERSON", "PERSON", "ORG", "PRIORITY", "CATEGORY", "DATE", "LOC"],
                "values": {
                    "IS_URGENT": lambda p: p == "high",
                    "STATUS": "pending"
                }
            },
            {
                "text": "{person}, a {title} in the {department} department, completed the {money} budget review with {sentiment} feedback.",
                "fields": ["PERSON", "TITLE", "DEPARTMENT", "MONEY", "SENTIMENT"],
                "values": {
                    "STATUS": "completed",
                    "REQUIRES_FOLLOWUP": False
                }
            }
        ]

    def generate_instruction(self, fields: List[str], detailed: bool = True) -> str:
        """Generate instruction with schema specification"""

        # Group fields by type
        extract_fields = []
        enum_fields = []
        boolean_fields = []

        for field in fields:
            if field in self.schema:
                field_info = self.schema[field]
                if field_info["type"] == FieldType.EXTRACT:
                    extract_fields.append(field)
                elif field_info["type"] == FieldType.ENUM:
                    enum_fields.append(field)
                elif field_info["type"] == FieldType.BOOLEAN:
                    boolean_fields.append(field)

        instruction_parts = ["Extract structured data according to this schema:"]

        # Add extract fields
        if extract_fields:
            for field in extract_fields:
                desc = self.schema[field]["description"]
                if detailed:
                    instruction_parts.append(f"- {field}: extract {desc} from text")
                else:
                    instruction_parts.append(f"- {field} (extract)")

        # Add enum fields with choices
        if enum_fields:
            for field in enum_fields:
                choices = self.schema[field]["choices"]
                desc = self.schema[field]["description"]
                if detailed:
                    if len(choices) <= 5:
                        instruction_parts.append(f"- {field}: select {desc} from [{', '.join(choices)}]")
                    else:
                        sample_choices = random.sample(choices, 5)
                        instruction_parts.append(f"- {field}: select {desc} from [{', '.join(sample_choices)}, ...]")
                else:
                    instruction_parts.append(f"- {field}: one of [{', '.join(choices[:3])}, ...]")

        # Add boolean fields
        if boolean_fields:
            for field in boolean_fields:
                desc = self.schema[field]["description"]
                if detailed:
                    instruction_parts.append(f"- {field}: boolean {desc}")
                else:
                    instruction_parts.append(f"- {field} (true/false)")

        instruction_parts.append("\nReturn as JSON with exact field names.")

        return "\n".join(instruction_parts)

    def generate_example(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a text example with structured output"""

        # Select a template
        template_data = random.choice(self.templates)
        template = template_data["text"]
        fields = template_data["fields"]
        predefined_values = template_data.get("values", {})

        # Generate text and collect values
        replacements = {}
        structured_output = {}

        # Track persons and orgs for uniqueness
        person_counter = 0

        for field in fields:
            if field == "PERSON":
                # Handle multiple persons
                person_counter += 1
                # First person is just "person", subsequent are "person2", "person3", etc
                key = "person" if person_counter == 1 else f"person{person_counter}"
                value = random.choice(self.data_pools["persons"])
                replacements[key] = value

                # Add to structured output
                if "PERSON" not in structured_output:
                    structured_output["PERSON"] = []
                structured_output["PERSON"].append(value)

            elif field == "ORG":
                value = random.choice(self.data_pools["organizations"])
                replacements["org"] = value
                if "ORG" not in structured_output:
                    structured_output["ORG"] = []
                structured_output["ORG"].append(value)

            elif field == "LOC":
                value = random.choice(self.data_pools["locations"])
                replacements["location"] = value
                if "LOC" not in structured_output:
                    structured_output["LOC"] = []
                structured_output["LOC"].append(value)

            elif field == "DATE":
                value = random.choice(self.data_pools["dates"])
                replacements["date"] = value
                if "DATE" not in structured_output:
                    structured_output["DATE"] = []
                structured_output["DATE"].append(value)

            elif field == "MONEY":
                value = random.choice(self.data_pools["money"])
                replacements["money"] = value
                if "MONEY" not in structured_output:
                    structured_output["MONEY"] = []
                structured_output["MONEY"].append(value)

            elif field in self.schema and self.schema[field]["type"] == FieldType.ENUM:
                # Handle enum fields
                value = random.choice(self.schema[field]["choices"])
                replacements[field.lower()] = value
                structured_output[field] = value

        # Generate text
        text = template
        for key, value in replacements.items():
            text = text.replace(f"{{{key}}}", str(value))

        # Add predefined values
        for field, value_gen in predefined_values.items():
            if callable(value_gen):
                # Dynamic value based on other fields
                args = []
                # Check if the function expects arguments
                if field == "IS_URGENT" and "priority" in replacements:
                    value = value_gen(replacements["priority"])
                elif field == "REQUIRES_FOLLOWUP" and "sentiment" in replacements:
                    value = value_gen(replacements["sentiment"])
                else:
                    value = value_gen()
            else:
                value = value_gen

            structured_output[field] = value

        # Ensure all requested fields are in output
        all_fields = set(fields + list(predefined_values.keys()))
        for field in all_fields:
            if field not in structured_output and field in self.schema:
                # Add default values for missing fields
                field_info = self.schema[field]
                if field_info["type"] == FieldType.ENUM:
                    structured_output[field] = random.choice(field_info["choices"])
                elif field_info["type"] == FieldType.BOOLEAN:
                    structured_output[field] = random.choice([True, False])

        return text, structured_output

    def generate_dataset(self, num_samples: int, instruction_style: str = "detailed") -> List[Dict[str, Any]]:
        """Generate complete dataset with structured output examples"""

        dataset = []

        for _ in range(num_samples):
            text, structured_output = self.generate_example()

            # Get relevant fields from the output
            fields = list(structured_output.keys())

            # Generate instruction
            detailed = instruction_style == "detailed" or (instruction_style == "mixed" and random.random() > 0.5)
            instruction = self.generate_instruction(fields, detailed=detailed)

            # Create conversation format
            messages = [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nText: {text}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps(structured_output, ensure_ascii=False)
                }
            ]

            dataset.append({"messages": messages})

        return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate structured output training data")
    parser.add_argument("--output", type=str, default="./data",
                       help="Output directory for generated data")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--instruction-style", type=str, default="mixed",
                       choices=["detailed", "simple", "mixed"],
                       help="Style of schema instructions")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training set ratio")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Create generator
    generator = StructuredOutputGenerator(seed=args.seed)

    # Generate dataset
    print(f"Generating {args.num_samples} structured output samples...")
    print(f"Instruction style: {args.instruction_style}")

    dataset = generator.generate_dataset(
        num_samples=args.num_samples,
        instruction_style=args.instruction_style
    )

    # Split dataset
    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))

    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]

    # Save datasets
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    processor = NERDataProcessor(use_conversation_format=True)

    # Save splits
    if train_data:
        train_file = output_path / "train.jsonl"
        processor.save_jsonl(train_data, train_file)
        print(f"Saved {len(train_data)} training samples to {train_file}")

    if val_data:
        val_file = output_path / "validation.jsonl"
        processor.save_jsonl(val_data, val_file)
        print(f"Saved {len(val_data)} validation samples to {val_file}")

    if test_data:
        test_file = output_path / "test.jsonl"
        processor.save_jsonl(test_data, test_file)
        print(f"Saved {len(test_data)} test samples to {test_file}")

    # Print sample
    print("\n" + "="*60)
    print("Sample structured output:")
    print("="*60)
    if dataset:
        sample = dataset[0]
        print(json.dumps(sample, ensure_ascii=False, indent=2))

    print("\n✓ Structured data generation completed!")
    print(f"Total samples: {len(dataset)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    main()