#!/bin/bash

echo "=================================================="
echo "vLLM Structured Output Usage Examples"
echo "=================================================="
echo ""

echo "✅ FIXED: vLLM API 수정 완료"
echo "- GuidedDecodingParams 사용 (StructuredOutputsParams 아님)"
echo "- json 파라미터 사용 (json_schema 아님)"
echo "- backend 지정하지 않음 (자동 선택)"
echo ""

echo "1. 단일 텍스트 처리:"
echo "--------------------"
echo "uv run python inference_vllm_structured.py \\"
echo "  --model-path ./outputs/quick_test/merged \\"
echo "  --text '김철수 대표가 삼성전자와 100억원 규모의 계약을 체결했습니다.' \\"
echo "  --schema-file test_schema.json"
echo ""

echo "2. 배치 처리:"
echo "------------"
echo "uv run python inference_vllm_structured.py \\"
echo "  --model-path ./outputs/quick_test/merged \\"
echo "  --input-file test_texts.txt \\"
echo "  --output-file results.json \\"
echo "  --schema-file test_schema.json \\"
echo "  --batch-size 4"
echo ""

echo "3. 스키마 없이 처리 (자유 생성):"
echo "------------------------------"
echo "uv run python inference_vllm_structured.py \\"
echo "  --model-path ./outputs/quick_test/merged \\"
echo "  --text '텍스트 내용' \\"
echo "  --no-guided-generation"
echo ""

echo "4. 커스텀 스키마 예제:"
echo "--------------------"
echo "cat > custom_schema.json << 'EOF'"
echo '['
echo '  {'
echo '    "name": "entities",'
echo '    "type": "extract",'
echo '    "description": "Extract all named entities"'
echo '  },'
echo '  {'
echo '    "name": "category",'
echo '    "type": "enum",'
echo '    "choices": ["news", "blog", "paper", "other"]'
echo '  },'
echo '  {'
echo '    "name": "is_important",'
echo '    "type": "boolean"'
echo '  }'
echo ']'
echo "EOF"
echo ""

echo "5. 테스트 실행:"
echo "--------------"
echo "# 간단한 테스트"
echo "uv run python test_simple_vllm.py"
echo ""
echo "# 전체 테스트 스위트"
echo "uv run python test_vllm_inference.py --model Qwen/Qwen2.5-3B-Instruct"
echo ""

echo "=================================================="
echo "주요 변경사항:"
echo "- vllm.sampling_params.GuidedDecodingParams 사용"
echo "- json 파라미터로 스키마 전달"
echo "- backend 자동 선택 (지정하지 않음)"
echo "=================================================="