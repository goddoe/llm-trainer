#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import NERDataProcessor
from src.model_config import (
    ModelConfig,
    SFTConfig,
    DataConfig,
    get_model_configs,
)


def setup_model_and_tokenizer(model_config: ModelConfig):
    print(f"Loading model: {model_config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
        use_auth_token=model_config.use_auth_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": model_config.trust_remote_code,
        "use_auth_token": model_config.use_auth_token,
        "device_map": model_config.device_map,
    }

    # Set appropriate dtype
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        **model_kwargs
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def create_data_collator(tokenizer, response_template: str = "Entities:"):
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    return collator


def main():
    parser = argparse.ArgumentParser(description="Train NER model with SFT")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model-type", type=str, default="gemma",
                       choices=["gemma", "llama", "mistral", "phi", "qwen"],
                       help="Type of base model to use")
    parser.add_argument("--data-path", type=str, help="Path to JSONL data")
    parser.add_argument("--dataset-name", type=str, default="conll2003",
                       help="Name of opensource dataset to use")
    parser.add_argument("--output-dir", type=str, default="./outputs/sft",
                       help="Output directory for model")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--use-completion-only", action="store_true",
                       help="Only train on completion/response part")

    args = parser.parse_args()

    # Load configurations
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model_config = ModelConfig(**config.get('model', {}))
        sft_config = SFTConfig(**config.get('training', {}))
        data_config = DataConfig(**config.get('data', {}))
    else:
        model_config = get_model_configs(args.model_type)
        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_length,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        data_config = DataConfig(
            train_file=args.data_path,
            dataset_name=args.dataset_name if not args.data_path else None,
        )
        model_config.max_length = args.max_length

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_config)

    # Enable gradient checkpointing if specified
    if sft_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare data
    data_processor = NERDataProcessor(max_length=model_config.max_length)

    if data_config.train_file:
        # Load from JSONL files
        dataset = data_processor.create_dataset_from_jsonl(
            train_path=data_config.train_file,
            val_path=data_config.validation_file,
        )
    else:
        # Load opensource dataset
        dataset = data_processor.load_opensource_dataset(data_config.dataset_name)

    # Prepare dataset for SFT
    if 'text' not in dataset['train'].column_names:
        dataset = dataset.map(
            data_processor.prepare_for_sft,
            batched=True,
            num_proc=sft_config.dataset_num_proc,
        )

    # Setup data collator
    if args.use_completion_only:
        data_collator = create_data_collator(tokenizer)
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    # Setup training arguments
    training_args = sft_config.to_training_args()

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get('train'),
        eval_dataset=dataset.get('validation') or dataset.get('test'),
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=sft_config.max_seq_length,
        packing=sft_config.packing,
        data_collator=data_collator if args.use_completion_only else None,
    )

    # Train
    print("Starting SFT training...")
    print(f"Training on {len(dataset['train'])} examples")
    if dataset.get('validation'):
        print(f"Validating on {len(dataset['validation'])} examples")

    trainer.train()

    # Save model
    print(f"Saving model to {sft_config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(sft_config.output_dir)

    # Evaluate if validation set exists
    if dataset.get('validation') or dataset.get('test'):
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")

        # Save evaluation results
        import json
        with open(f"{sft_config.output_dir}/eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)

    print("SFT training completed successfully!")


if __name__ == "__main__":
    main()