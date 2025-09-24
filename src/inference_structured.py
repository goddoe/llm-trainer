#!/usr/bin/env python3
"""
Inference script for structured output with the trained model.
Supports both regular inference and schema-constrained generation.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


class FieldType(Enum):
    """Types of fields in structured output"""
    EXTRACT = "extract"  # Extract from text
    ENUM = "enum"  # Select from predefined choices
    BOOLEAN = "boolean"  # True/False


@dataclass
class FieldSchema:
    """Schema for a single field"""
    name: str
    type: FieldType
    description: str
    choices: Optional[List[str]] = None  # For ENUM type


class StructuredInference:
    """Inference with structured output support"""

    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        """Initialize the inference engine"""
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        if base_model:
            # Load base model with LoRA adapter
            print(f"Loading base model: {base_model}")

            # Quantization config
            bnb_config = None
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif load_in_8bit:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            print(f"Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)

            # Use tokenizer from adapter if available, otherwise from base model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            # Load merged model directly
            print(f"Loading model from: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def create_prompt(
        self,
        text: str,
        schema: Optional[List[FieldSchema]] = None
    ) -> str:
        """Create prompt with schema instructions"""
        if schema:
            # Build structured prompt with schema
            instruction = "Extract information from the following text and return it in JSON format.\n\n"
            instruction += "Required fields:\n"

            for field in schema:
                if field.type == FieldType.EXTRACT:
                    instruction += f"- {field.name}: {field.description} (extract from text)\n"
                elif field.type == FieldType.ENUM:
                    choices_str = ", ".join([f'"{c}"' for c in field.choices])
                    instruction += f"- {field.name}: {field.description} (choose from: {choices_str})\n"
                elif field.type == FieldType.BOOLEAN:
                    instruction += f"- {field.name}: {field.description} (true or false)\n"

            instruction += f"\nText: {text}"
        else:
            # Default NER prompt
            instruction = f"Extract named entities from the following text and return them in JSON format.\n\nText: {text}"

        # Apply chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": instruction}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = instruction

        return prompt

    def generate(
        self,
        text: str,
        schema: Optional[List[FieldSchema]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        """Generate structured output"""
        prompt = self.create_prompt(text, schema)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        # Try to parse as JSON
        try:
            result = json.loads(generated)

            # Validate against schema if provided
            if schema:
                validated_result = self.validate_schema(result, schema)
                return validated_result

            return result
        except json.JSONDecodeError:
            # Return raw text if not valid JSON
            return {"raw_output": generated, "error": "Failed to parse JSON"}

    def validate_schema(
        self,
        result: Dict[str, Any],
        schema: List[FieldSchema]
    ) -> Dict[str, Any]:
        """Validate and clean result against schema"""
        validated = {}

        for field in schema:
            value = result.get(field.name)

            if field.type == FieldType.ENUM and field.choices:
                # Ensure value is from choices
                if value not in field.choices:
                    # Try to find closest match
                    if value and isinstance(value, str):
                        value_lower = value.lower()
                        for choice in field.choices:
                            if choice.lower() == value_lower:
                                value = choice
                                break
                    if value not in field.choices:
                        value = field.choices[0]  # Default to first choice

            elif field.type == FieldType.BOOLEAN:
                # Ensure boolean
                if isinstance(value, str):
                    value = value.lower() in ["true", "yes", "1"]
                elif not isinstance(value, bool):
                    value = bool(value) if value else False

            elif field.type == FieldType.EXTRACT:
                # Keep as is for extraction fields
                if value is None:
                    value = []  # Default to empty list for extraction

            validated[field.name] = value

        return validated


def main():
    parser = argparse.ArgumentParser(description="Structured inference with trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model or LoRA adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model name if using LoRA adapter"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to process"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="File with texts to process (one per line)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for results"
    )
    parser.add_argument(
        "--schema-file",
        type=str,
        help="JSON file with field schema"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization"
    )

    args = parser.parse_args()

    # Load schema if provided
    schema = None
    if args.schema_file:
        with open(args.schema_file) as f:
            schema_data = json.load(f)
            schema = []
            for field_data in schema_data:
                schema.append(FieldSchema(
                    name=field_data["name"],
                    type=FieldType(field_data["type"]),
                    description=field_data.get("description", ""),
                    choices=field_data.get("choices")
                ))

    # Initialize inference engine
    engine = StructuredInference(
        model_path=args.model_path,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Process text(s)
    if args.text:
        # Single text
        result = engine.generate(
            args.text,
            schema=schema,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        print("\nInput:", args.text)
        print("Output:", json.dumps(result, indent=2))

    elif args.input_file:
        # Batch processing
        results = []
        with open(args.input_file) as f:
            texts = [line.strip() for line in f if line.strip()]

        for text in texts:
            print(f"\nProcessing: {text[:100]}...")
            result = engine.generate(
                text,
                schema=schema,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            results.append({
                "input": text,
                "output": result
            })

        # Save results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output_file}")
        else:
            print("\nResults:")
            for r in results:
                print(json.dumps(r, indent=2))
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            text = input("\nEnter text: ")
            if text.lower() == "quit":
                break

            result = engine.generate(
                text,
                schema=schema,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            print("Output:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()