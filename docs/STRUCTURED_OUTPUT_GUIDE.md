# Structured Output Training and Inference Guide

## Overview

This system enables training language models to generate structured JSON output with mixed constraint types:
- **EXTRACT**: Open-ended extraction from text (e.g., names, locations)
- **ENUM**: Constrained selection from predefined choices (e.g., sentiment: positive/negative/neutral)
- **BOOLEAN**: True/false fields (e.g., contains_financial_info: true/false)

## Key Features

### 1. Training System
- **Conversation Format**: Uses instruction-tuned model format with user/assistant messages
- **TRL v0.23 Integration**: Leverages latest TRL SFTTrainer with conversation support
- **LoRA Support**: Efficient parameter-efficient fine-tuning
- **Multiple Model Support**: Qwen3, Gemma-3, Llama, Mistral, Phi

### 2. Data Generation
- **Structured Data Generator** (`scripts/generate_structured_data.py`):
  - Generates training data with mixed field types
  - Creates realistic scenarios with schema compliance
  - Supports diverse domains and instruction styles

### 3. Inference Options

#### Standard Inference (`inference_structured.py`)
- HuggingFace Transformers-based
- Schema validation and type coercion
- Supports both LoRA adapters and merged models
- Batch processing capability

#### vLLM Inference (`inference_vllm_structured.py`)
- High-performance serving with vLLM
- **Guided Generation**: JSON schema enforcement during generation
- Tensor parallelism support
- Efficient batch processing

## Quick Start

### 1. Generate Training Data
```bash
# Generate 1000 structured samples
uv run python scripts/generate_structured_data.py \
  --output ./data/structured \
  --num-samples 1000
```

### 2. Train Model
```bash
# LoRA training (recommended for quick iteration)
uv run python -m src.train \
  --use-lora \
  --model-type qwen3 \
  --data-path ./data/structured/train.jsonl \
  --use-conversation-format \
  --num-epochs 3
```

### 3. Run Inference

#### Option A: Regular Inference
```bash
uv run python inference_structured.py \
  --model-path ./outputs/lora/merged \
  --text "Your text here" \
  --schema-file schema.json
```

#### Option B: vLLM with Guided Generation
```bash
uv run python inference_vllm_structured.py \
  --model-path ./outputs/lora/merged \
  --text "Your text here" \
  --schema-file schema.json
```

## Schema Definition

Create a JSON schema file defining your output structure:

```json
[
  {
    "name": "people",
    "type": "extract",
    "description": "Names of people mentioned"
  },
  {
    "name": "sentiment",
    "type": "enum",
    "description": "Overall sentiment",
    "choices": ["positive", "negative", "neutral"]
  },
  {
    "name": "urgent",
    "type": "boolean",
    "description": "Whether the message is urgent"
  }
]
```

## Training Results

The 5-step quick training demonstration showed:
- Loss decreased from 2.61 to 2.36
- Token accuracy improved to ~60%
- Model learned basic conversation format
- For production: train for 3-5 epochs with 1000+ samples

## Key Differences from Regular NER

1. **Structured Output**: Returns validated JSON with defined schema
2. **Mixed Constraints**: Combines extraction with constrained choices
3. **Type Safety**: Enforces boolean, enum, and array types
4. **Guided Generation**: vLLM can enforce schema during generation
5. **Conversation Format**: Uses chat templates for instruction-tuned models

## Configuration Files

- `configs/qwen3_conversation.yaml`: Qwen3 model configuration
- `configs/gemma3_conversation.yaml`: Gemma-3 model configuration
- `configs/quick_test.yaml`: 5-step test configuration

## Files Created

### Core Scripts
- `inference_structured.py`: HuggingFace-based inference with schema support
- `inference_vllm_structured.py`: vLLM inference with guided generation
- `scripts/generate_structured_data.py`: Training data generator

### Example Files
- `test_schema.json`: Example schema definition
- `test_texts.txt`: Sample texts for testing
- `example_usage.sh`: Usage examples for all scripts

## Performance Tips

1. **Training**:
   - Use LoRA for faster iteration
   - Start with smaller models (Qwen3-0.6B, Gemma-270M)
   - Use conversation format for instruction-tuned models

2. **Inference**:
   - Use vLLM for production serving
   - Enable guided generation for guaranteed schema compliance
   - Batch requests for better throughput

## Next Steps

1. Train with more data (1000+ samples) for better performance
2. Experiment with different schemas for your use case
3. Fine-tune temperature and sampling parameters
4. Consider using larger base models for complex tasks