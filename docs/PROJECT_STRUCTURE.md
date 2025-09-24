# Project Structure

## Directory Layout

```
llm-trainer/
├── src/                    # Core source code
│   ├── train.py           # Main training script
│   ├── inference.py       # Inference engine
│   ├── data_processor.py  # Data processing utilities
│   └── model_config.py    # Configuration classes
│
├── scripts/               # Utility scripts
│   ├── generate_structured_data.py
│   ├── generate_conversation_data.py
│   ├── convert_to_conversation.py
│   ├── validate_conversation_data.py
│   └── prepare_data.py
│
├── configs/               # Configuration files
│   ├── quick_test.yaml   # 5-step test config
│   ├── qwen3_conversation.yaml
│   └── gemma3_conversation.yaml
│
├── tests/                 # Test files
│   ├── test_vllm_inference.py
│   ├── test_simple_vllm.py
│   ├── test_conversation_format.py
│   ├── test_instruction_styles.py
│   ├── demo_structured_output.py
│   ├── test_schema.json   # Test schema
│   └── test_texts.txt     # Test texts
│
├── examples/              # Example scripts
│   ├── example_usage.sh
│   ├── example_conversation_training.sh
│   └── vllm_usage_examples.sh
│
├── docs/                  # Documentation
│   ├── STRUCTURED_OUTPUT_GUIDE.md
│   └── NER_LABELING_DATA_REFERENCE.md
│
├── data/                  # Data files
│   ├── 20250911 NER 라벨링 데이터/
│   └── [various training data]
│
├── outputs/               # Model outputs
│   └── quick_test/       # Test training results
│
├── logs/                  # Log files
│   └── training_output.log
│
├── kb_docs/              # Knowledge base docs
│   └── trl_sft_usage.md
│
├── inference_structured.py       # HuggingFace inference
├── inference_vllm_structured.py  # vLLM inference
├── README.md             # Project README
├── CLAUDE.md            # Claude Code instructions
├── PROJECT_STRUCTURE.md # This file
└── pyproject.toml       # Project configuration
```

## Key Files

### Training
- `src/train.py` - Main training script with LoRA/SFT support
- `configs/*.yaml` - Training configurations

### Inference
- `inference_structured.py` - Standard HuggingFace inference
- `inference_vllm_structured.py` - vLLM with guided generation

### Data Generation
- `scripts/generate_structured_data.py` - Create structured output data
- `scripts/generate_conversation_data.py` - Create conversation format data

### Testing
- `tests/test_vllm_inference.py` - Comprehensive vLLM tests
- `tests/test_simple_vllm.py` - Simple vLLM test

## Usage

### Training
```bash
uv run python -m src.train --config configs/quick_test.yaml
```

### Inference
```bash
# HuggingFace
uv run python inference_structured.py \
  --model-path ./outputs/quick_test/merged \
  --text "Your text" \
  --schema-file tests/test_schema.json

# vLLM
uv run python inference_vllm_structured.py \
  --model-path ./outputs/quick_test/merged \
  --text "Your text" \
  --schema-file tests/test_schema.json
```

### Testing
```bash
# Run tests
uv run python tests/test_vllm_inference.py

# Run examples
bash examples/example_usage.sh
```

## Model Support
- Qwen3 (Qwen/Qwen3-0.6B)
- Gemma-3 (google/gemma-3-270m-it)
- Llama (meta-llama/Llama-3.2-1B)
- Mistral (mistralai/Mistral-7B-v0.1)
- Phi (microsoft/phi-2)