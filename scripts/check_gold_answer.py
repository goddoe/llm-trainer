#!/usr/bin/env python3
"""
gold_answer_nl 컬럼을 정확하게 확인
"""

import csv
import json
import glob
from collections import defaultdict, Counter

def check_gold_answer_nl():
    """gold_answer_nl 컬럼을 정확히 확인"""

    csv_files = glob.glob("./data/**/*.csv", recursive=True)

    print(f"Found {len(csv_files)} CSV files")
    print("="*60)

    all_keys = set()
    all_values = defaultdict(list)
    files_with_data = 0
    total_rows_with_data = 0
    sample_data = []

    for csv_file in csv_files[:30]:  # 30개 파일 확인
        print(f"\nChecking: {csv_file.split('/')[-1]}")
        has_data_in_file = False

        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    # gold_answer_nl 컬럼 확인
                    gold_nl = row.get('gold_answer_nl', '').strip()

                    # 빈 값이 아니고 {} 도 아닌 경우
                    if gold_nl and gold_nl != '{}' and gold_nl != '{ }':
                        try:
                            data = json.loads(gold_nl)

                            if data and isinstance(data, dict) and len(data) > 0:
                                has_data_in_file = True
                                total_rows_with_data += 1

                                # 샘플 저장
                                if len(sample_data) < 10:
                                    sample_data.append({
                                        'file': csv_file.split('/')[-1],
                                        'row': i+1,
                                        'sentence': row.get('sentence', '')[:100],
                                        'gold_answer_nl': data
                                    })

                                # 키 수집
                                for key in data.keys():
                                    all_keys.add(key)

                                # 값 수집
                                for key, value in data.items():
                                    if isinstance(value, list):
                                        all_values[key].extend(value)
                                    else:
                                        all_values[key].append(value)

                                # 첫 몇 개 출력
                                if total_rows_with_data <= 5:
                                    print(f"  Row {i+1}: {json.dumps(data, ensure_ascii=False)}")

                        except json.JSONDecodeError as e:
                            print(f"  Row {i+1}: JSON decode error - {gold_nl[:50]}...")
                        except Exception as e:
                            print(f"  Row {i+1}: Error - {e}")

        except Exception as e:
            print(f"  Error reading file: {e}")

        if has_data_in_file:
            files_with_data += 1
            print(f"  ✅ Found data in this file")
        else:
            print(f"  ❌ No data in gold_answer_nl")

    # 결과 출력
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files with data: {files_with_data}/{len(csv_files[:30])}")
    print(f"Total rows with data: {total_rows_with_data}")
    print(f"Unique keys found: {len(all_keys)}")

    if all_keys:
        print("\n🔑 All Keys Found:")
        for key in sorted(all_keys):
            value_count = len(set(all_values[key]))
            sample_values = list(set(all_values[key]))[:5]
            print(f"  - {key} ({value_count} unique values)")
            if sample_values:
                print(f"    Examples: {sample_values}")

    # 샘플 데이터 출력
    if sample_data:
        print("\n" + "="*60)
        print("SAMPLE DATA")
        print("="*60)

        for i, sample in enumerate(sample_data[:5], 1):
            print(f"\n[Sample {i}]")
            print(f"File: {sample['file']}")
            print(f"Row: {sample['row']}")
            print(f"Text: {sample['sentence']}")
            print(f"Gold: {json.dumps(sample['gold_answer_nl'], ensure_ascii=False, indent=2)}")

    # 결과 저장
    result = {
        'files_checked': len(csv_files[:30]),
        'files_with_data': files_with_data,
        'total_rows_with_data': total_rows_with_data,
        'all_keys': sorted(list(all_keys)),
        'key_value_samples': {
            key: {
                'unique_count': len(set(all_values[key])),
                'samples': list(set(all_values[key]))[:20]
            }
            for key in all_keys
        },
        'sample_data': sample_data
    }

    with open('gold_answer_check.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Detailed results saved to: gold_answer_check.json")

if __name__ == "__main__":
    check_gold_answer_nl()