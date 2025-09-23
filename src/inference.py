import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel


class NERInference:
    def __init__(
        self,
        model_path: str,
        base_model_name: Optional[str] = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_length: int = 512,
    ):
        self.model_path = Path(model_path)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Check if it's a LoRA model
        adapter_config_path = self.model_path / "adapter_config.json"
        self.is_lora = adapter_config_path.exists()

        if self.is_lora and not base_model_name:
            # Try to read base model from adapter config
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")
                if not base_model_name:
                    raise ValueError("Base model name required for LoRA model")

        self.model, self.tokenizer = self._load_model_and_tokenizer(
            base_model_name, load_in_8bit, load_in_4bit
        )

        # Setup generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _load_model_and_tokenizer(
        self,
        base_model_name: Optional[str],
        load_in_8bit: bool,
        load_in_4bit: bool
    ):
        if self.is_lora:
            # Load base model first
            print(f"Loading base model: {base_model_name}")

            model_kwargs = {"device_map": "auto" if self.device == "auto" else None}

            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                model_kwargs["torch_dtype"] = torch.float16

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **model_kwargs
            )

            # Load LoRA adapter
            print(f"Loading LoRA adapter from: {self.model_path}")
            model = PeftModel.from_pretrained(base_model, str(self.model_path))

            # Load tokenizer from LoRA path or base model
            tokenizer_path = self.model_path if (self.model_path / "tokenizer_config.json").exists() else base_model_name
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            # Load complete model
            print(f"Loading model from: {self.model_path}")

            model_kwargs = {"device_map": "auto" if self.device == "auto" else None}

            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                model_kwargs["torch_dtype"] = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def format_input(self, text: str, add_instruction: bool = True) -> str:
        if add_instruction:
            prompt = f"Extract named entities from the following text and return them in JSON format.\n\nText: {text}\n\nEntities:"
        else:
            prompt = f"Text: {text}\n\nEntities:"
        return prompt

    def extract_entities(
        self,
        text: str,
        add_instruction: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> Dict[str, List[str]]:
        # Format input
        prompt = self.format_input(text, add_instruction)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        if self.device != "auto" and self.device != "cpu":
            inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part
        if "Entities:" in generated_text:
            response = generated_text.split("Entities:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()

        # Parse JSON
        try:
            entities = json.loads(response)
            if not isinstance(entities, dict):
                entities = {}
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    entities = json.loads(json_match.group())
                except:
                    entities = {}
            else:
                entities = {}

        return entities

    def batch_extract_entities(
        self,
        texts: List[str],
        batch_size: int = 8,
        add_instruction: bool = True,
    ) -> List[Dict[str, List[str]]]:
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for text in batch:
                entities = self.extract_entities(text, add_instruction)
                results.append(entities)

        return results

    def evaluate_on_test_data(
        self,
        test_file: str,
        output_file: Optional[str] = None
    ) -> Dict:
        from src.data_processor import NERDataProcessor
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score

        processor = NERDataProcessor()
        test_data = processor.load_jsonl(test_file)

        predictions = []
        gold_labels = []
        correct = 0
        total = 0

        for item in test_data:
            text = item['input']
            gold_entities = json.loads(item['output']) if isinstance(item['output'], str) else item['output']

            # Get prediction
            pred_entities = self.extract_entities(text)

            predictions.append(pred_entities)
            gold_labels.append(gold_entities)

            # Calculate exact match accuracy
            if pred_entities == gold_entities:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        results = {
            "accuracy": accuracy,
            "total_examples": total,
            "correct_predictions": correct,
        }

        # Save predictions if output file specified
        if output_file:
            output_data = []
            for i, (pred, gold, item) in enumerate(zip(predictions, gold_labels, test_data)):
                output_data.append({
                    "id": i,
                    "input": item['input'],
                    "gold": gold,
                    "prediction": pred,
                    "correct": pred == gold,
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"Predictions saved to {output_file}")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run NER inference")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--base-model", type=str,
                       help="Base model name (required for LoRA models)")
    parser.add_argument("--input", type=str,
                       help="Input text to extract entities from")
    parser.add_argument("--input-file", type=str,
                       help="File containing texts (one per line)")
    parser.add_argument("--test-file", type=str,
                       help="JSONL test file for evaluation")
    parser.add_argument("--output", type=str,
                       help="Output file for predictions")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Load model in 8-bit precision")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load model in 4-bit precision")
    parser.add_argument("--no-instruction", action="store_true",
                       help="Don't add instruction to prompt")

    args = parser.parse_args()

    # Initialize inference
    inference = NERInference(
        model_path=args.model_path,
        base_model_name=args.base_model,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    if args.test_file:
        # Evaluate on test file
        results = inference.evaluate_on_test_data(args.test_file, args.output)
        print(f"Evaluation Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Correct: {results['correct_predictions']}/{results['total_examples']}")

    elif args.input_file:
        # Process file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        results = inference.batch_extract_entities(
            texts,
            add_instruction=not args.no_instruction
        )

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for text, entities in zip(texts, results):
                    f.write(json.dumps({
                        "input": text,
                        "entities": entities
                    }, ensure_ascii=False) + '\n')
            print(f"Results saved to {args.output}")
        else:
            for text, entities in zip(texts, results):
                print(f"Text: {text}")
                print(f"Entities: {json.dumps(entities, ensure_ascii=False, indent=2)}")
                print("-" * 50)

    elif args.input:
        # Process single text
        entities = inference.extract_entities(
            args.input,
            add_instruction=not args.no_instruction
        )
        print(f"Input: {args.input}")
        print(f"Entities: {json.dumps(entities, ensure_ascii=False, indent=2)}")

    else:
        # Interactive mode
        print("Interactive NER extraction mode. Type 'quit' to exit.")
        while True:
            text = input("\nEnter text: ").strip()
            if text.lower() == 'quit':
                break

            entities = inference.extract_entities(
                text,
                add_instruction=not args.no_instruction
            )
            print(f"Entities: {json.dumps(entities, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()