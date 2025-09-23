from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTConfig as TRLSFTConfig


@dataclass
class ModelConfig:
    model_name: str = "google/gemma-2b"
    use_auth_token: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: str = "auto"
    trust_remote_code: bool = True
    max_length: int = 512
    use_flash_attention: bool = False


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM

    def to_peft_config(self) -> LoraConfig:
        target_modules = self.target_modules
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"]

        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
        )


@dataclass
class SFTConfig:
    max_seq_length: int = 512
    packing: bool = False
    dataset_text_field: str = "text"
    dataset_num_proc: int = 4
    num_train_epochs: int = 3
    max_steps: int = -1  # Support for limiting training steps
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # Support for warmup steps
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    output_dir: str = "./outputs"
    overwrite_output_dir: bool = True
    bf16: bool = False
    fp16: bool = False
    tf32: bool = True
    optim: str = "adamw_torch"
    seed: int = 42
    remove_unused_columns: bool = False

    # Conversation format settings
    use_conversation_format: bool = True
    assistant_only_loss: bool = True  # Only compute loss on assistant responses
    completion_only_loss: bool = False  # For prompt-completion format

    # Chat template settings
    chat_template: Optional[str] = None  # Custom chat template
    tokenizer_chat_template: bool = True  # Use tokenizer's built-in template

    def to_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=self.overwrite_output_dir,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            push_to_hub=self.push_to_hub,
            hub_model_id=self.hub_model_id,
            report_to=self.report_to,
            bf16=self.bf16,
            fp16=self.fp16,
            tf32=self.tf32,
            optim=self.optim,
            seed=self.seed,
            remove_unused_columns=self.remove_unused_columns,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    def to_trl_sft_config(self) -> TRLSFTConfig:
        """Convert to TRL's SFTConfig for conversation training"""
        return TRLSFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            bf16=self.bf16,
            tf32=self.tf32,
            optim=self.optim,
            seed=self.seed,
            remove_unused_columns=self.remove_unused_columns,
            report_to=self.report_to,
            push_to_hub=self.push_to_hub,
            hub_model_id=self.hub_model_id,
            max_length=self.max_seq_length,  # Changed from max_seq_length to max_length
            packing=self.packing,
            dataset_num_proc=self.dataset_num_proc,
        )


@dataclass
class DataConfig:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    dataset_name: Optional[str] = "conll2003"
    max_samples: Optional[int] = None
    validation_split: float = 0.1
    seed: int = 42


def get_model_configs(model_type: str = "gemma") -> ModelConfig:
    configs = {
        "gemma": ModelConfig(
            model_name="google/gemma-2b",
            max_length=512,
        ),
        "gemma-3-it": ModelConfig(
            model_name="google/gemma-3-270m-it",
            max_length=512,
        ),
        "llama": ModelConfig(
            model_name="meta-llama/Llama-3.2-1B",
            max_length=512,
        ),
        "mistral": ModelConfig(
            model_name="mistralai/Mistral-7B-v0.1",
            max_length=512,
            load_in_4bit=True,
        ),
        "phi": ModelConfig(
            model_name="microsoft/phi-2",
            max_length=512,
        ),
        "qwen": ModelConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            max_length=512,
        ),
        "qwen3": ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            max_length=512,
        ),
    }

    return configs.get(model_type, configs["gemma"])


def get_lora_config(base_model: str = "gemma") -> LoRAConfig:
    configs = {
        "gemma": LoRAConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
        ),
        "llama": LoRAConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
        ),
        "mistral": LoRAConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
        ),
        "phi": LoRAConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "dense",
                          "fc1", "fc2"],
            lora_dropout=0.1,
        ),
        "qwen": LoRAConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
        ),
    }

    return configs.get(base_model, configs["gemma"])