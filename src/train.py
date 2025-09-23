#!/usr/bin/env python3
"""
Unified training script for NER models with LoRA and SFT support
Combines both parameter-efficient (LoRA) and full fine-tuning (SFT) in one script
"""
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
    BitsAndBytesConfig,
)
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import json

from data_processor import NERDataProcessor
from model_config import (
    ModelConfig,
    LoRAConfig,
    SFTConfig,
    DataConfig,
    get_model_configs,
    get_lora_config
)


def setup_model_and_tokenizer(model_config: ModelConfig, use_quantization: bool = False):
    """Setup model and tokenizer with optional quantization"""
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

    # Configure quantization if requested
    if use_quantization or model_config.load_in_8bit or model_config.load_in_4bit:
        if model_config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
        elif model_config.load_in_4bit:
            try:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print("Warning: bitsandbytes not available, skipping 4-bit quantization")
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        # Set appropriate dtype based on hardware
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        **model_kwargs
    )

    # Prepare for k-bit training if using quantization
    if model_config.load_in_8bit or model_config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def create_data_collator(tokenizer, use_completion_only: bool = False):
    """Create appropriate data collator"""
    if use_completion_only:
        response_template = "Entities:"
        response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
            mlm=False,
        )
    else:
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Unified NER model training with LoRA/SFT")

    # Training mode selection
    parser.add_argument("--use-lora", action="store_true",
                       help="Use LoRA for parameter-efficient training")
    parser.add_argument("--config", type=str, help="Path to config file")

    # Model configuration
    parser.add_argument("--model-type", type=str, default="gemma",
                       choices=["gemma", "llama", "mistral", "phi", "qwen"],
                       help="Type of base model to use")
    parser.add_argument("--model-name", type=str,
                       help="Custom model name/path (overrides model-type)")

    # Data configuration
    parser.add_argument("--data-path", type=str, help="Path to JSONL data")
    parser.add_argument("--dataset-name", type=str, default="conll2003",
                       help="Name of opensource dataset to use")

    # Training configuration
    parser.add_argument("--output-dir", type=str, default="./outputs/unified",
                       help="Output directory for model")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")

    # LoRA specific arguments
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--lora-target-modules", nargs="+",
                       help="Target modules for LoRA")

    # Optimization arguments
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Load model in 8-bit precision")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load model in 4-bit precision")
    parser.add_argument("--use-completion-only", action="store_true",
                       help="Only train on completion/response part")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true",
                       help="Use BF16 training")

    # Other arguments
    parser.add_argument("--push-to-hub", action="store_true",
                       help="Push model to HuggingFace Hub")
    parser.add_argument("--hub-model-id", type=str,
                       help="HuggingFace Hub model ID")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum number of training samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Load configurations
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Extract use_lora from config if not specified in command line
        if 'training' in config and 'use_lora' in config['training'] and not args.use_lora:
            args.use_lora = config['training']['use_lora']

        model_config = ModelConfig(**config.get('model', {}))
        lora_config = LoRAConfig(**config.get('lora', {})) if args.use_lora else None
        sft_config = SFTConfig(**config.get('training', {}))
        data_config = DataConfig(**config.get('data', {}))
    else:
        # Build configs from command line arguments
        if args.model_name:
            model_config = ModelConfig(
                model_name=args.model_name,
                max_length=args.max_length,
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
            )
        else:
            model_config = get_model_configs(args.model_type)
            model_config.max_length = args.max_length
            model_config.load_in_8bit = args.load_in_8bit
            model_config.load_in_4bit = args.load_in_4bit

        if args.use_lora:
            lora_config = get_lora_config(args.model_type)
            lora_config.r = args.lora_r
            lora_config.lora_alpha = args.lora_alpha
            lora_config.lora_dropout = args.lora_dropout
            if args.lora_target_modules:
                lora_config.target_modules = args.lora_target_modules
        else:
            lora_config = None

        # Adjust default learning rate based on training type
        default_lr = args.learning_rate if args.learning_rate != 2e-4 else (
            2e-4 if args.use_lora else 5e-5
        )

        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=default_lr,
            max_seq_length=args.max_length,
            gradient_checkpointing=args.gradient_checkpointing,
            fp16=args.fp16,
            bf16=args.bf16,
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )

        data_config = DataConfig(
            train_file=args.data_path,
            dataset_name=args.dataset_name if not args.data_path else None,
            max_samples=args.max_samples,
            seed=args.seed,
        )

    # Print training mode
    print("=" * 50)
    print(f"Training Mode: {'LoRA (Parameter-Efficient)' if args.use_lora else 'SFT (Full Fine-Tuning)'}")
    print(f"Model: {model_config.model_name}")
    print(f"Output Directory: {sft_config.output_dir}")
    print("=" * 50)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_config,
        use_quantization=(args.load_in_8bit or args.load_in_4bit)
    )

    # Apply LoRA if requested
    if args.use_lora:
        print("\nApplying LoRA configuration...")
        peft_config = lora_config.to_peft_config()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("\nUsing full model for SFT...")
        if sft_config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Prepare data
    print("\nPreparing dataset...")
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

    # Apply max_samples if specified
    if data_config.max_samples:
        for split in dataset.keys():
            if len(dataset[split]) > data_config.max_samples:
                dataset[split] = dataset[split].select(range(data_config.max_samples))

    # Prepare dataset for SFT
    if 'text' not in dataset['train'].column_names:
        dataset = dataset.map(
            data_processor.prepare_for_sft,
            batched=True,
            num_proc=sft_config.dataset_num_proc,
        )

    print(f"Training samples: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"Validation samples: {len(dataset['validation'])}")
    elif 'test' in dataset:
        print(f"Test samples: {len(dataset['test'])}")

    # Setup data collator
    data_collator = create_data_collator(tokenizer, args.use_completion_only)

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
    print("\nStarting training...")
    trainer.train()

    # Save model
    print(f"\nSaving model to {sft_config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(sft_config.output_dir)

    # Save training configuration
    config_to_save = {
        "training_mode": "lora" if args.use_lora else "sft",
        "model_name": model_config.model_name,
        "training_args": {
            "num_epochs": sft_config.num_train_epochs,
            "batch_size": sft_config.per_device_train_batch_size,
            "learning_rate": sft_config.learning_rate,
            "max_length": sft_config.max_seq_length,
        }
    }

    if args.use_lora:
        config_to_save["lora_config"] = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
        }

        # Save merged model if using LoRA
        if hasattr(model, 'merge_and_unload'):
            print("\nMerging and saving LoRA weights...")
            merged_model = model.merge_and_unload()
            merged_output_dir = f"{sft_config.output_dir}/merged"
            merged_model.save_pretrained(merged_output_dir)
            tokenizer.save_pretrained(merged_output_dir)
            print(f"Saved merged model to {merged_output_dir}")

    with open(f"{sft_config.output_dir}/training_config.json", 'w') as f:
        json.dump(config_to_save, f, indent=2)

    # Evaluate if validation set exists
    if dataset.get('validation') or dataset.get('test'):
        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")

        # Save evaluation results
        with open(f"{sft_config.output_dir}/eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Training completed successfully!")
    print(f"Model saved to: {sft_config.output_dir}")
    if args.use_lora:
        print(f"Merged model saved to: {sft_config.output_dir}/merged")
    print("=" * 50)


if __name__ == "__main__":
    main()