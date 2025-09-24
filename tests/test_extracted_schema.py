#!/usr/bin/env python3
"""
기계적으로 추출한 스키마로 테스트 수행
"""

import json
import subprocess
import sys
import glob
import csv
import random
from pathlib import Path

def get_sample_texts(num_samples=10):
    """CSV에서 샘플 텍스트 추출"""
    texts = []
    csv_files = glob.glob("./data/**/*.csv", recursive=True)[:5]

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sentence = row.get('sentence', '').strip()
                    if sentence and len(sentence) > 10:
                        texts.append(sentence)
                        if len(texts) >= num_samples:
                            return texts
        except Exception as e:
            continue

    return texts

def test_huggingface_inference():
    """HuggingFace 추론 테스트"""
    print("\n" + "="*60)
    print("HuggingFace Inference Test with Extracted Schema")
    print("="*60)

    # 샘플 텍스트 가져오기
    texts = get_sample_texts(5)

    if not texts:
        print("❌ No sample texts found")
        return False

    results = []

    for i, text in enumerate(texts, 1):
        print(f"\n[Test {i}]")
        print(f"Input: {text[:100]}...")

        cmd = [
            "uv", "run", "python", "src/inference_structured.py",
            "--model-path", "./outputs/quick_test/merged",
            "--text", text,
            "--schema-file", "data/schemas/extracted_entities_schema.json",
            "--temperature", "0.1",
            "--max-new-tokens", "200"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # 결과 파싱
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if 'Output:' in line:
                    json_start = line.find('{')
                    if json_start != -1:
                        try:
                            output_json = json.loads(line[json_start:])
                            print(f"✅ Success: {json.dumps(output_json, ensure_ascii=False)}")
                            results.append(True)
                        except json.JSONDecodeError:
                            print(f"⚠️ Invalid JSON: {line[json_start:json_start+50]}...")
                            results.append(False)
                    break

        except subprocess.TimeoutExpired:
            print("❌ Timeout")
            results.append(False)
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(False)

    success_rate = sum(results) / len(results) * 100 if results else 0
    print(f"\n📊 Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")

    return success_rate > 0

def test_vllm_inference():
    """vLLM 추론 테스트"""
    print("\n" + "="*60)
    print("vLLM Inference Test with Extracted Schema")
    print("="*60)

    # 샘플 텍스트 가져오기
    texts = get_sample_texts(3)

    if not texts:
        print("❌ No sample texts found")
        return False

    # 텍스트 파일로 저장
    with open('test_texts_extracted.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    cmd = [
        "uv", "run", "python", "src/inference_vllm_structured.py",
        "--model-path", "./outputs/quick_test/merged",
        "--input-file", "test_texts_extracted.txt",
        "--output-file", "test_results_extracted.json",
        "--schema-file", "data/schemas/gold_answer_schema.json",
        "--batch-size", "3",
        "--max-tokens", "8192"
    ]

    print("Running vLLM batch inference...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and Path("test_results_extracted.json").exists():
            with open("test_results_extracted.json", 'r', encoding='utf-8') as f:
                results = json.load(f)

            print(f"\n✅ Processed {len(results)} texts")

            for i, item in enumerate(results, 1):
                print(f"\n[Result {i}]")
                print(f"Input: {item['input'][:80]}...")
                if isinstance(item['output'], dict):
                    print(f"Output: {json.dumps(item['output'], ensure_ascii=False, indent=2)}")
                else:
                    print(f"Output: {item['output']}")

            return True
        else:
            print(f"❌ vLLM failed: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Timeout (120s)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def analyze_schema():
    """스키마 분석 및 통계"""
    print("\n" + "="*60)
    print("Schema Analysis")
    print("="*60)

    # 스키마 로드 - gold_answer_schema 사용
    schema_file = "data/schemas/gold_answer_schema.json" if Path("data/schemas/gold_answer_schema.json").exists() else "data/schemas/extracted_entities_schema.json"
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    # 분석 결과 로드 - gold_answer_analysis 사용
    analysis_file = "data/schemas/gold_answer_analysis.json" if Path("data/schemas/gold_answer_analysis.json").exists() else "data/schemas/extracted_entities_analysis.json"
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    # Coverage 정보 확인
    if 'sentences_with_entities' in analysis:
        print(f"\n📊 Coverage: {analysis['sentences_with_entities']}/{analysis['total_sentences']} sentences ({analysis['sentences_with_entities']/analysis['total_sentences']*100:.1f}%)")
    elif 'rows_with_data' in analysis:
        print(f"\n📊 Coverage: {analysis['rows_with_data']}/{analysis['total_rows']} rows ({analysis['rows_with_data']/analysis['total_rows']*100:.1f}%)")

    print(f"\n🔑 Entity Types: {len(schema)}")

    extract_types = [s for s in schema if s['type'] == 'extract']
    enum_types = [s for s in schema if s['type'] == 'enum']

    print(f"  - Extract types: {len(extract_types)}")
    print(f"  - Enum types: {len(enum_types)}")

    print("\n📈 Top Entities by Frequency:")

    # gold_answer_analysis 형식 확인
    if 'key_details' in analysis:
        # gold_answer_analysis 형식
        for key in ['환자 증상1', '환자발생 유형_질병_병력', '의식상태_1차', '활력징후_1차_맥박', '주증상(1개선택)']:
            if key in analysis['key_details']:
                entity_data = analysis['key_details'][key]
                freq = entity_data['frequency']
                unique = entity_data['unique_values']
                print(f"  {key}: {freq}회, {unique}개 고유값")
    elif 'entity_types' in analysis:
        # extracted_entities_analysis 형식
        for entity_name in ['신체부위', '증상', '장소', '시간', '인물']:
            if entity_name in analysis['entity_types']:
                entity_data = analysis['entity_types'][entity_name]
                top_3 = entity_data['top_10'][:3]
                top_3_str = ', '.join([f"{v[0]}({v[1]})" for v in top_3])
                print(f"  {entity_name}: {top_3_str}")

    return True

def validate_extraction():
    """추출된 엔티티 검증"""
    print("\n" + "="*60)
    print("Extraction Validation")
    print("="*60)

    # 테스트 문장들
    test_sentences = [
        "머리가 아프고 열이 나요",
        "어제부터 배가 아파서 병원에 갔어요",
        "할머니가 계단에서 넘어졌어요",
        "지금 의식이 있나요?",
        "산소마스크 필요해요"
    ]

    # 추출 스크립트 import
    import sys
    sys.path.append('scripts')
    from extract_entities_from_csv import extract_entities_from_text

    print("\n🔍 Testing entity extraction on sample sentences:")

    for sentence in test_sentences:
        entities = extract_entities_from_text(sentence)
        print(f"\nText: {sentence}")
        print(f"Extracted: {json.dumps(entities, ensure_ascii=False, indent=2)}")

    return True

def main():
    """메인 테스트 함수"""
    print("="*60)
    print("🧪 EXTRACTED SCHEMA TEST SUITE")
    print("="*60)

    # 필요 파일 확인
    required_files = [
        "data/schemas/extracted_entities_schema.json",
        "data/schemas/extracted_entities_analysis.json"
    ]

    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing required file: {file}")
            print("Please run: uv run python scripts/extract_entities_from_csv.py")
            return 1

    results = []

    # 1. 스키마 분석
    print("\n[1/4] Analyzing Schema...")
    results.append(analyze_schema())

    # 2. 추출 검증
    print("\n[2/4] Validating Extraction...")
    results.append(validate_extraction())

    # 3. HuggingFace 테스트
    print("\n[3/4] Testing HuggingFace Inference...")
    results.append(test_huggingface_inference())

    # 4. vLLM 테스트 (선택적)
    if "--with-vllm" in sys.argv:
        print("\n[4/4] Testing vLLM Inference...")
        results.append(test_vllm_inference())
    else:
        print("\n[4/4] Skipping vLLM test (use --with-vllm to enable)")

    # 최종 결과
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)

    test_names = ["Schema Analysis", "Extraction Validation", "HuggingFace Test", "vLLM Test"]
    for i, (name, result) in enumerate(zip(test_names[:len(results)], results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")

    success_rate = sum(results) / len(results) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")

    if success_rate < 50:
        print("\n💡 Tip: The model needs more training. Try:")
        print("  uv run python -m src.train --config configs/train_config.yaml --num-epochs 3")

    return 0 if success_rate >= 50 else 1

if __name__ == "__main__":
    sys.exit(main())