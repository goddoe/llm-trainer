#!/usr/bin/env python3
"""
Simple test to verify vLLM structured output is working
"""

import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


def test_basic_structured_output():
    """Test basic structured output with vLLM"""
    print("Testing vLLM Structured Output")
    print("="*50)

    # Initialize model
    model = "Qwen/Qwen2.5-3B-Instruct"  # Use a known good model
    llm = LLM(model=model, max_model_len=512, gpu_memory_utilization=0.5)

    # Define schema
    schema = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {"type": "string"}
            },
            "companies": {
                "type": "array",
                "items": {"type": "string"}
            },
            "amount": {"type": "string"}
        },
        "required": ["people", "companies", "amount"]
    }

    # Create prompt
    prompt = "Extract entities: 김철수 대표가 삼성전자와 100억원 규모의 계약을 체결했습니다."

    # Setup guided generation
    guided_params = GuidedDecodingParams(json=schema)
    sampling_params = SamplingParams(
        guided_decoding=guided_params,
        temperature=0.1,
        max_tokens=100
    )

    # Generate
    print(f"\nPrompt: {prompt}")
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    result = outputs[0].outputs[0].text

    # Parse and display
    try:
        parsed = json.loads(result)
        print(f"\nOutput: {json.dumps(parsed, indent=2, ensure_ascii=False)}")
        print("\n✅ Success! Structured output is working correctly.")
        return True
    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse JSON: {e}")
        print(f"Raw output: {result}")
        return False


if __name__ == "__main__":
    success = test_basic_structured_output()
    exit(0 if success else 1)