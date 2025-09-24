#!/usr/bin/env python3
"""
vLLM inference script with structured output enforcement.
Uses vLLM's guided generation with JSON schema validation.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


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
    required: bool = True


class VLLMStructuredInference:
    """vLLM-based inference with structured output enforcement"""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
    ):
        """Initialize vLLM engine"""
        print(f"Loading model with vLLM: {model_path}")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )

        # Try to get tokenizer from model
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            self.tokenizer = None

    def create_json_schema(self, schema: List[FieldSchema]) -> Dict[str, Any]:
        """Create JSON schema for structured output validation"""
        properties = {}
        required = []

        for field in schema:
            if field.type == FieldType.EXTRACT:
                # Array of strings for extraction
                properties[field.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": field.description
                }

            elif field.type == FieldType.ENUM:
                # Enum with specific choices
                properties[field.name] = {
                    "type": "string",
                    "enum": field.choices,
                    "description": field.description
                }

            elif field.type == FieldType.BOOLEAN:
                # Boolean field
                properties[field.name] = {
                    "type": "boolean",
                    "description": field.description
                }

            if field.required:
                required.append(field.name)

        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }

        return json_schema

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
        if self.tokenizer and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
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
        texts: List[str],
        schema: Optional[List[FieldSchema]] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 256,
        use_guided_generation: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate structured output for batch of texts"""
        # Create prompts
        prompts = [self.create_prompt(text, schema) for text in texts]

        # Setup sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # Add guided generation if schema provided
        if schema and use_guided_generation:
            json_schema = self.create_json_schema(schema)
            # Use vLLM GuidedDecodingParams with correct parameter name
            try:
                guided_params = GuidedDecodingParams(
                    json=json_schema  # Use 'json' instead of 'json_schema', no backend
                )
                sampling_params.guided_decoding = guided_params
            except Exception as e:
                print(f"Warning: Could not apply guided decoding: {e}")
                # Fallback for older versions
                try:
                    sampling_params.guided_json = json_schema
                except:
                    pass

        # Generate
        outputs = self.llm.generate(prompts, sampling_params)

        # Parse results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text

            try:
                # Try to parse as JSON
                result = json.loads(generated_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to clean up truncated JSON
                try:
                    # Try to fix common truncation issues
                    if generated_text.count('{') > generated_text.count('}'):
                        # Add missing closing braces
                        missing_braces = generated_text.count('{') - generated_text.count('}')
                        generated_text_fixed = generated_text + ('}' * missing_braces)
                        result = json.loads(generated_text_fixed)
                    else:
                        result = {"raw_output": generated_text, "error": "Failed to parse JSON"}
                except:
                    # Store raw output without Unicode escaping
                    result = {"raw_output": generated_text, "error": "Failed to parse JSON"}

            results.append(result)

        return results

    def generate_single(
        self,
        text: str,
        schema: Optional[List[FieldSchema]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate for a single text"""
        results = self.generate([text], schema, **kwargs)
        return results[0] if results else {}


def main():
    parser = argparse.ArgumentParser(description="vLLM structured inference")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model"
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
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization"
    )
    parser.add_argument(
        "--no-guided-generation",
        action="store_true",
        help="Disable guided generation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
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
                    choices=field_data.get("choices"),
                    required=field_data.get("required", True)
                ))

    # Initialize vLLM engine
    engine = VLLMStructuredInference(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Process text(s)
    if args.text:
        # Single text
        result = engine.generate_single(
            args.text,
            schema=schema,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            use_guided_generation=not args.no_guided_generation,
        )
        print("\nInput:", args.text)
        print("Output:", json.dumps(result, indent=2, ensure_ascii=False))

    elif args.input_file:
        # Batch processing
        with open(args.input_file) as f:
            texts = [line.strip() for line in f if line.strip()]

        all_results = []
        # Process in batches
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(texts)-1)//args.batch_size + 1}...")

            batch_results = engine.generate(
                batch_texts,
                schema=schema,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                use_guided_generation=not args.no_guided_generation,
            )

            for text, result in zip(batch_texts, batch_results):
                all_results.append({
                    "input": text,
                    "output": result
                })

        # Save results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output_file}")
        else:
            print("\nResults:")
            for r in all_results:
                print(json.dumps(r, indent=2, ensure_ascii=False))

    else:
        print("Please provide --text or --input-file")


if __name__ == "__main__":
    main()
