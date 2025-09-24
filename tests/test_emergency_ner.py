#!/usr/bin/env python3
"""
Test emergency NER with structured output schema
"""

import json
import csv
import random
from pathlib import Path

def extract_test_texts_from_csv(csv_dir: str, num_samples: int = 10):
    """Extract sample texts from CSV files"""
    texts = []

    # Find CSV files using glob
    import glob
    csv_files = glob.glob(f"{csv_dir}/*.csv") or glob.glob("./data/**/*.csv", recursive=True)[:10]

    for csv_file in csv_files[:5]:  # Sample from first 5 files
        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sentence = row.get('sentence', '').strip()
                    if sentence and len(sentence) > 10:  # Skip very short sentences
                        texts.append(sentence)
                        if len(texts) >= num_samples:
                            return texts
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    return texts

def test_with_huggingface():
    """Test with HuggingFace inference"""
    print("\n" + "="*60)
    print("Testing with HuggingFace Inference")
    print("="*60)

    # Get sample texts
    csv_dir = "./data/20250911 NER 라벨링 데이터/label/라벨링 작업 파일/all_completed_exports_20250916_163050"
    texts = extract_test_texts_from_csv(csv_dir, num_samples=5)

    if not texts:
        print("No texts found in CSV files")
        return

    # Test each text
    for i, text in enumerate(texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")

        # Run inference
        import subprocess
        cmd = [
            "uv", "run", "python", "inference_structured.py",
            "--model-path", "./outputs/quick_test/merged",
            "--text", text,
            "--schema-file", "emergency_ner_schema.json",
            "--temperature", "0.1"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Parse output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.startswith('Output:'):
                    # Find the JSON part
                    json_start = line.find('{')
                    if json_start != -1:
                        json_str = line[json_start:]
                        try:
                            output = json.loads(json_str)
                            print(f"Extracted entities: {json.dumps(output, ensure_ascii=False, indent=2)}")
                        except:
                            print(f"Raw output: {line}")
                    break
        except subprocess.TimeoutExpired:
            print("  Timeout!")
        except Exception as e:
            print(f"  Error: {e}")

def test_with_vllm():
    """Test with vLLM inference"""
    print("\n" + "="*60)
    print("Testing with vLLM Inference")
    print("="*60)

    # Get sample texts
    csv_dir = "./data/20250911 NER 라벨링 데이터/label/라벨링 작업 파일/all_completed_exports_20250916_163050"
    texts = extract_test_texts_from_csv(csv_dir, num_samples=3)

    if not texts:
        print("No texts found in CSV files")
        return

    # Save texts to file for batch processing
    with open('emergency_test_texts.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    # Run vLLM batch inference
    import subprocess
    cmd = [
        "uv", "run", "python", "inference_vllm_structured.py",
        "--model-path", "./outputs/quick_test/merged",
        "--input-file", "emergency_test_texts.txt",
        "--output-file", "emergency_results.json",
        "--schema-file", "emergency_ner_schema.json",
        "--batch-size", "3"
    ]

    print("Running vLLM batch inference...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            # Load and display results
            try:
                with open('emergency_results.json', 'r', encoding='utf-8') as f:
                    results = json.load(f)

                for i, item in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Input: {item['input'][:100]}...")
                    print(f"Output: {json.dumps(item['output'], ensure_ascii=False, indent=2)}")
            except Exception as e:
                print(f"Error reading results: {e}")
        else:
            print(f"Error running vLLM: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("Timeout!")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main test function"""
    print("Emergency NER Schema Test")
    print("Schema: emergency_ner_schema.json")

    # Check if schema exists
    if not Path("emergency_ner_schema.json").exists():
        print("Error: emergency_ner_schema.json not found!")
        return

    # Display schema
    with open("emergency_ner_schema.json", 'r', encoding='utf-8') as f:
        schema = json.load(f)
        print(f"\nSchema has {len(schema)} entity types:")
        for item in schema[:5]:  # Show first 5
            print(f"  - {item['name']}: {item['type']} - {item['description']}")
        if len(schema) > 5:
            print(f"  ... and {len(schema)-5} more")

    # Test with HuggingFace
    test_with_huggingface()

    # Optional: Test with vLLM
    # test_with_vllm()

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    main()