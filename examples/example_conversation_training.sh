#!/bin/bash
# Example script showing how to use conversation format for training

echo "=========================================="
echo "Conversation Format Training Example"
echo "=========================================="

# Step 1: Generate conversation data
echo ""
echo "Step 1: Generating conversation training data..."
uv run python scripts/generate_conversation_data.py \
    --output ./data/conversation \
    --num-samples 500 \
    --style balanced \
    --seed 42

# Step 2: Validate the generated data
echo ""
echo "Step 2: Validating conversation data..."
uv run python scripts/validate_conversation_data.py ./data/conversation

# Step 3: Train with Qwen3 using conversation format
echo ""
echo "Step 3: Training with Qwen3-0.6B (conversation format)..."
echo "Command to run:"
echo "uv run python -m src.train --config configs/qwen3_conversation.yaml"

# Step 4: Train with Gemma-3-IT using conversation format
echo ""
echo "Step 4: Training with Gemma-3-270m-it (conversation format)..."
echo "Command to run:"
echo "uv run python -m src.train --config configs/gemma3_conversation.yaml"

# Alternative: Direct training command
echo ""
echo "Alternative: Direct training with conversation format:"
echo "uv run python -m src.train \\"
echo "    --use-lora \\"
echo "    --model-type qwen3 \\"
echo "    --data-path ./data/conversation/train.jsonl \\"
echo "    --use-conversation-format \\"
echo "    --assistant-only-loss \\"
echo "    --output-dir ./outputs/conversation-model"

echo ""
echo "=========================================="
echo "✓ Example setup complete!"
echo "Use the commands above to train with conversation format."
echo "=========================================="