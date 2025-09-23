#!/usr/bin/env python3
"""
Advanced conversation data generator for NER training
Generates natural, multi-turn conversations for instruction-tuned models
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.data_processor import NERDataProcessor


class ConversationGenerator:
    """Generate diverse conversation-style training data for NER"""

    def __init__(self, style: str = "balanced", seed: int = 42):
        self.style = style
        random.seed(seed)

        # Define possible entity types and their descriptions
        self.entity_types_info = {
            "PERSON": ["person names", "individuals", "people"],
            "ORG": ["organizations", "companies", "institutions", "agencies"],
            "LOC": ["locations", "places", "geographical locations", "cities", "countries"],
            "DATE": ["dates", "time references", "temporal expressions"],
            "MONEY": ["monetary values", "financial amounts", "currency"],
            "EVENT": ["events", "conferences", "meetings", "occasions"],
            "PRODUCT": ["products", "services", "software", "devices"],
            "PERCENT": ["percentages", "percentage values"],
            "TITLE": ["job titles", "positions", "roles"]
        }

        # Instruction templates with entity keys and JSON format instructions
        self.instruction_templates = {
            "formal": [
                "Extract the following entity types: PERSON, ORG, LOC, DATE, MONEY, EVENT, PRODUCT. Return the results in JSON format.",
                "Identify entities for these categories: PERSON (individuals), ORG (organizations/companies), LOC (locations), DATE (temporal expressions), MONEY (monetary values). Format output as JSON.",
                "Extract named entities with keys: PERSON, ORG, LOC, DATE, EVENT. Provide the output in JSON format with entity types as keys and lists of entities as values.",
                "Perform entity extraction for: PERSON (names of people), ORG (organizations), LOC (geographical locations), MONEY (financial amounts), PRODUCT (products/services). Return JSON formatted results.",
            ],
            "casual": [
                "Find entities for: PERSON, ORG, LOC. Return as JSON.",
                "Extract: people (PERSON), companies (ORG), places (LOC), dates (DATE). Give me the results in JSON format.",
                "Look for: PERSON (names), ORG (organizations), LOC (locations), MONEY (amounts). Output should be JSON.",
                "Get me: PERSON, ORG, LOC, DATE, EVENT entities. Format as JSON please.",
            ],
            "technical": [
                "Execute NER for entity types: {PERSON: [person names], ORG: [organizations], LOC: [locations], DATE: [dates], MONEY: [monetary values]}. Return structured JSON output.",
                "Apply entity recognition for keys: PERSON, ORG, LOC, DATE, MONEY, EVENT, PRODUCT. Output format: JSON with entity_type -> [entity_list] mapping.",
                "Extract entities with schema: {\"PERSON\": [], \"ORG\": [], \"LOC\": [], \"DATE\": [], \"MONEY\": []}. Populate and return as valid JSON.",
                "Perform extraction for types: [PERSON|ORG|LOC|DATE|MONEY|EVENT]. Return JSON object with entity types as keys.",
            ],
            "specific": [
                "Extract only: PERSON (individual names), ORG (company/organization names), LOC (city/country names). JSON format required.",
                "Find these specific entities: {\"PERSON\": [people], \"ORG\": [companies], \"MONEY\": [amounts], \"DATE\": [dates]}. Return as JSON.",
                "Identify: PERSON names like 'John Smith', ORG names like 'Google', LOC names like 'New York'. Output in JSON format.",
                "Extract entities matching: PERSON (e.g., 'Emma Wilson'), ORG (e.g., 'Microsoft'), LOC (e.g., 'London'), DATE (e.g., 'January 2024'). Format as JSON.",
            ]
        }


        # Entity data for generation
        self.entity_data = {
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
            ],
            "percentages": [
                "25%", "50%", "12.5%", "100%", "3.7%", "87%", "0.5%", "33%"
            ],
            "products": [
                "iPhone 15", "GPT-4", "Model S", "Surface Pro", "Pixel 8",
                "ChatGPT", "Windows 11", "MacBook Pro", "Quest 3", "Gemini"
            ],
            "events": [
                "World Cup", "Olympics", "CES 2024", "WWDC", "Climate Summit",
                "G20 Summit", "Tech Conference", "Annual Meeting", "Product Launch"
            ]
        }

        # Context templates for more complex scenarios
        self.context_templates = [
            {
                "context": "In a recent financial report,",
                "template": "{org} announced {financial_metric} in {date}, exceeding analyst expectations by {percentage}.",
                "entities": ["org", "money", "date", "percentage"]
            },
            {
                "context": "During the technology conference,",
                "template": "{person}, CEO of {org}, unveiled {product} at {event} in {location}.",
                "entities": ["person", "org", "product", "event", "location"]
            },
            {
                "context": "According to industry sources,",
                "template": "{org1} is planning to acquire {org2} for {money}, pending regulatory approval in {location}.",
                "entities": ["org", "org", "money", "location"]
            },
            {
                "context": "In an exclusive interview,",
                "template": "{person1} and {person2} discussed the future of {org} and its expansion plans in {location}.",
                "entities": ["person", "person", "org", "location"]
            },
            {
                "context": "Breaking news:",
                "template": "The {event} will be held in {location} on {date}, featuring keynote speakers from {org1} and {org2}.",
                "entities": ["event", "location", "date", "org", "org"]
            }
        ]

    def get_instruction(self, style: str = None, entity_types: List[str] = None) -> str:
        """Get a random instruction based on style, optionally customized for specific entity types"""
        style = style or self.style

        # If specific entity types are provided, create a custom instruction
        if entity_types:
            entity_descriptions = []
            for etype in entity_types:
                if etype in self.entity_types_info:
                    desc = random.choice(self.entity_types_info[etype])
                    entity_descriptions.append(f"{etype} ({desc})")
                else:
                    entity_descriptions.append(etype)

            entity_list = ", ".join(entity_descriptions[:3]) + (" and more" if len(entity_descriptions) > 3 else "")

            custom_templates = [
                f"Extract the following entities: {', '.join(entity_types)}. Return results in JSON format.",
                f"Find: {entity_list}. Output as JSON with entity types as keys.",
                f"Identify entities for types: {', '.join(entity_types)}. Format the output as JSON.",
            ]
            return random.choice(custom_templates)

        # Otherwise use predefined templates
        if style == "balanced":
            # Mix all styles
            all_instructions = []
            for templates in self.instruction_templates.values():
                all_instructions.extend(templates)
            return random.choice(all_instructions)
        else:
            return random.choice(self.instruction_templates.get(style, self.instruction_templates["formal"]))

    def generate_simple_example(self) -> Tuple[str, Dict[str, List[str]]]:
        """Generate a simple single-sentence example"""
        templates = [
            ("{person} works at {org} in {location}.", {
                "PERSON": ["{person}"], "ORG": ["{org}"], "LOC": ["{location}"]
            }),
            ("{org} hired {person} as their new {title}.", {
                "ORG": ["{org}"], "PERSON": ["{person}"], "TITLE": ["{title}"]
            }),
            ("The {event} will take place on {date} in {location}.", {
                "EVENT": ["{event}"], "DATE": ["{date}"], "LOC": ["{location}"]
            }),
            ("{person} from {org} announced {money} in funding.", {
                "PERSON": ["{person}"], "ORG": ["{org}"], "MONEY": ["{money}"]
            }),
            ("{org1} partnered with {org2} to develop {product}.", {
                "ORG": ["{org1}", "{org2}"], "PRODUCT": ["{product}"]
            }),
        ]

        template_text, template_entities = random.choice(templates)

        # Create replacements
        replacements = {}
        replacements["person"] = random.choice(self.entity_data["persons"])
        replacements["org"] = random.choice(self.entity_data["organizations"])
        replacements["org1"] = random.choice(self.entity_data["organizations"])
        replacements["org2"] = random.choice([o for o in self.entity_data["organizations"] if o != replacements.get("org1", "")])
        replacements["location"] = random.choice(self.entity_data["locations"])
        replacements["date"] = random.choice(self.entity_data["dates"])
        replacements["money"] = random.choice(self.entity_data["money"])
        replacements["event"] = random.choice(self.entity_data["events"])
        replacements["product"] = random.choice(self.entity_data["products"])
        replacements["title"] = random.choice(["CEO", "CTO", "CFO", "Director", "Manager", "Engineer"])
        replacements["percentage"] = random.choice(self.entity_data["percentages"])

        # Fill template
        text = template_text
        entities = {}

        for placeholder, value in replacements.items():
            text = text.replace(f"{{{placeholder}}}", value)

        for entity_type, entity_list in template_entities.items():
            entities[entity_type] = []
            for entity_template in entity_list:
                for placeholder, value in replacements.items():
                    entity_template = entity_template.replace(f"{{{placeholder}}}", value)
                if entity_template not in entities[entity_type]:
                    entities[entity_type].append(entity_template)

        return text, entities

    def generate_complex_example(self) -> Tuple[str, Dict[str, List[str]]]:
        """Generate a complex multi-sentence example with context"""
        context_item = random.choice(self.context_templates)

        # Build the text
        context = context_item["context"]
        template = context_item["template"]

        # Create replacements
        replacements = {}
        entity_counts = {}

        for entity_key in context_item["entities"]:
            # Handle multiple entities of same type (org1, org2, etc.)
            base_key = entity_key.rstrip("0123456789")

            if base_key == "person":
                value = random.choice(self.entity_data["persons"])
            elif base_key == "org":
                # Ensure different orgs if multiple
                if base_key in entity_counts:
                    existing = [v for k, v in replacements.items() if k.startswith(base_key)]
                    available = [o for o in self.entity_data["organizations"] if o not in existing]
                    value = random.choice(available) if available else random.choice(self.entity_data["organizations"])
                else:
                    value = random.choice(self.entity_data["organizations"])
                entity_counts[base_key] = entity_counts.get(base_key, 0) + 1
            elif entity_key == "location":
                value = random.choice(self.entity_data["locations"])
            elif entity_key == "date":
                value = random.choice(self.entity_data["dates"])
            elif entity_key == "money":
                value = random.choice(self.entity_data["money"])
            elif entity_key == "event":
                value = random.choice(self.entity_data["events"])
            elif entity_key == "product":
                value = random.choice(self.entity_data["products"])
            elif entity_key == "percentage":
                value = random.choice(self.entity_data["percentages"])
            else:
                continue

            # Store with potential number suffix
            if base_key in entity_counts and entity_counts[base_key] > 1:
                replacements[f"{base_key}{entity_counts[base_key]}"] = value
            else:
                replacements[entity_key] = value

        # Build the full text
        text = context + " " + template

        # Replace placeholders
        for key, value in replacements.items():
            text = text.replace(f"{{{key}}}", value)

        # Build entities dict
        entities = {}
        type_mapping = {
            "person": "PERSON",
            "org": "ORG",
            "location": "LOC",
            "date": "DATE",
            "money": "MONEY",
            "event": "EVENT",
            "product": "PRODUCT",
            "percentage": "PERCENT"
        }

        for key, value in replacements.items():
            base_key = key.rstrip("0123456789")
            entity_type = type_mapping.get(base_key, base_key.upper())

            if entity_type not in entities:
                entities[entity_type] = []
            if value not in entities[entity_type]:
                entities[entity_type].append(value)

        # Add some additional context sentences occasionally
        if random.random() > 0.5:
            additional_sentences = [
                f"This marks a significant milestone for the industry.",
                f"Analysts predict continued growth in the coming quarters.",
                f"The announcement was well-received by stakeholders.",
                f"Further details will be announced in the coming weeks.",
            ]
            text += " " + random.choice(additional_sentences)

        return text, entities


    def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate a complete dataset with specified number of samples"""
        dataset = []

        for i in range(num_samples):
            # Generate single-turn conversation only
            if random.random() > 0.3:
                text, entities = self.generate_complex_example()
            else:
                text, entities = self.generate_simple_example()

            # Get instruction, sometimes customized for the actual entity types in the example
            if random.random() > 0.5 and entities:
                # Use actual entity types from the example
                instruction = self.get_instruction(entity_types=list(entities.keys()))
            else:
                # Use general instruction
                instruction = self.get_instruction()
            messages = [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nText: {text}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps(entities, ensure_ascii=False, indent=2)
                }
            ]
            dataset.append({"messages": messages})

        return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate conversation-style NER training data")
    parser.add_argument("--output", type=str, default="./data",
                       help="Output directory for generated data")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Total number of samples to generate")
    parser.add_argument("--style", type=str, default="balanced",
                       choices=["formal", "casual", "technical", "specific", "balanced"],
                       help="Instruction style for conversations")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training set ratio")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--include-simple", action="store_true",
                       help="Include more simple examples")

    args = parser.parse_args()

    # Create generator
    generator = ConversationGenerator(style=args.style, seed=args.seed)

    # Generate dataset
    print(f"Generating {args.num_samples} conversation samples...")
    print(f"Style: {args.style}")

    dataset = generator.generate_dataset(
        num_samples=args.num_samples
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

    # Save train set
    if train_data:
        train_file = output_path / "train.jsonl"
        processor.save_jsonl(train_data, train_file)
        print(f"Saved {len(train_data)} training samples to {train_file}")

    # Save validation set
    if val_data:
        val_file = output_path / "validation.jsonl"
        processor.save_jsonl(val_data, val_file)
        print(f"Saved {len(val_data)} validation samples to {val_file}")

    # Save test set
    if test_data:
        test_file = output_path / "test.jsonl"
        processor.save_jsonl(test_data, test_file)
        print(f"Saved {len(test_data)} test samples to {test_file}")

    # Print sample
    print("\n" + "="*50)
    print("Sample generated conversation:")
    print("="*50)
    sample = dataset[0]
    print(json.dumps(sample, ensure_ascii=False, indent=2))

    print("\n✓ Conversation data generation completed!")
    print(f"Total samples: {len(dataset)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    main()