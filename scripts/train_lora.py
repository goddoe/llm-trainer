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
)
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import NERDataProcessor
from src.model_config import (
    ModelConfig,
    LoRAConfig,
    SFTConfig,
    DataConfig,
    get_model_configs,
    get_lora_config
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

    if model_config.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["torch_dtype"] = torch.float16
    elif model_config.load_in_4bit:
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
        model_config.model_name,
        **model_kwargs
    )

    if model_config.load_in_8bit or model_config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train NER model with LoRA")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model-type", type=str, default="gemma",
                       choices=["gemma", "llama", "mistral", "phi", "qwen"],
                       help="Type of base model to use")
    parser.add_argument("--data-path", type=str, help="Path to JSONL data")
    parser.add_argument("--dataset-name", type=str, default="conll2003",
                       help="Name of opensource dataset to use")
    parser.add_argument("--output-dir", type=str, default="./outputs/lora",
                       help="Output directory for model")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")

    args = parser.parse_args()

    # Load configurations
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model_config = ModelConfig(**config.get('model', {}))
        lora_config = LoRAConfig(**config.get('lora', {}))
        sft_config = SFTConfig(**config.get('training', {}))
        data_config = DataConfig(**config.get('data', {}))
    else:
        model_config = get_model_configs(args.model_type)
        lora_config = get_lora_config(args.model_type)
        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_length,
        )
        data_config = DataConfig(
            train_file=args.data_path,
            dataset_name=args.dataset_name if not args.data_path else None,
        )

        lora_config.r = args.lora_r
        lora_config.lora_alpha = args.lora_alpha
        model_config.max_length = args.max_length

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_config)

    # Apply LoRA
    peft_config = lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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

    # Setup trainer
    training_args = sft_config.to_training_args()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get('train'),
        eval_dataset=dataset.get('validation') or dataset.get('test'),
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=sft_config.max_seq_length,
        packing=sft_config.packing,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {sft_config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(sft_config.output_dir)

    # Save final model with merged LoRA weights
    if hasattr(model, 'merge_and_unload'):
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(f"{sft_config.output_dir}/merged")
        print(f"Saved merged model to {sft_config.output_dir}/merged")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()