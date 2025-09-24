#!/usr/bin/env python3
"""
gold_answer_nl 컬럼을 정확하게 파싱 (멀티라인 JSON 처리)
"""

import csv
import json
import glob
import re
from collections import defaultdict, Counter

def clean_and_parse_json(text):
    """멀티라인 JSON 문자열을 정리하고 파싱"""
    if not text or text.strip() in ['{}', '{ }', '']:
        return None

    # 줄바꿈과 공백 정리
    text = text.strip()

    # 이미 유효한 JSON인지 확인
    try:
        return json.loads(text)
    except:
        pass

    # 줄바꿈 처리 - JSON 내부의 줄바꿈 보존
    # '{' 다음에 오는 줄바꿈과 공백 정리
    text = re.sub(r'\{\s+', '{', text)
    text = re.sub(r'\s+\}', '}', text)
    text = re.sub(r'\[\s+', '[', text)
    text = re.sub(r'\s+\]', ']', text)

    # 줄바꿈을 공백으로 (문자열 내부 제외)
    lines = []
    in_string = False
    for char in text:
        if char == '"' and (len(lines) == 0 or lines[-1] != '\\'):
            in_string = not in_string
        if char == '\n' and not in_string:
            lines.append(' ')
        else:
            lines.append(char)

    text = ''.join(lines)

    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)

    try:
        return json.loads(text)
    except Exception as e:
        # 마지막 시도 - eval (위험하지만 제한된 환경에서)
        try:
            # 작은 따옴표를 큰 따옴표로
            text = text.replace("'", '"')
            return json.loads(text)
        except:
            return None

def extract_gold_answer_nl():
    """gold_answer_nl에서 실제 키와 값 추출"""

    csv_files = glob.glob("./data/**/*.csv", recursive=True)[:30]

    print(f"Processing {len(csv_files)} CSV files...")
    print("="*60)

    all_keys = set()
    all_values = defaultdict(Counter)
    key_frequency = Counter()
    value_examples = defaultdict(set)
    sample_data = []
    total_rows = 0
    rows_with_data = 0

    for csv_file in csv_files:
        file_name = csv_file.split('/')[-1]
        print(f"\nProcessing: {file_name[:50]}...")

        try:
            # 파일 전체를 읽어서 처리
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                content = f.read()

            # CSV로 파싱
            import io
            reader = csv.DictReader(io.StringIO(content))

            for row_num, row in enumerate(reader, 1):
                total_rows += 1

                gold_nl = row.get('gold_answer_nl', '')

                if gold_nl and gold_nl.strip() not in ['{}', '{ }', '']:
                    # JSON 파싱 시도
                    data = clean_and_parse_json(gold_nl)

                    if data and isinstance(data, dict) and len(data) > 0:
                        rows_with_data += 1

                        # 첫 5개 샘플 저장
                        if len(sample_data) < 5:
                            sample_data.append({
                                'file': file_name,
                                'row': row_num,
                                'sentence': row.get('sentence', '')[:100],
                                'gold_answer_nl': data
                            })
                            print(f"  ✅ Row {row_num}: Found {len(data)} keys")
                            for key in list(data.keys())[:3]:
                                print(f"      - {key}: {data[key]}")

                        # 키 수집
                        for key in data.keys():
                            all_keys.add(key)
                            key_frequency[key] += 1

                        # 값 수집
                        for key, value in data.items():
                            if isinstance(value, list):
                                for item in value:
                                    all_values[key][str(item)] += 1
                                    value_examples[key].add(str(item))
                            else:
                                all_values[key][str(value)] += 1
                                value_examples[key].add(str(value))

        except Exception as e:
            print(f"  Error: {e}")

    # 결과 출력
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    print(f"Total rows: {total_rows}")
    print(f"Rows with gold_answer_nl data: {rows_with_data}")
    print(f"Coverage: {rows_with_data/total_rows*100:.1f}%" if total_rows > 0 else "N/A")
    print(f"Unique keys found: {len(all_keys)}")

    if all_keys:
        print("\n" + "="*60)
        print("ALL KEYS FOUND IN gold_answer_nl")
        print("="*60)

        for key in sorted(all_keys):
            freq = key_frequency[key]
            values = all_values[key]
            examples = sorted(list(value_examples[key]))[:5]

            print(f"\n📌 {key}")
            print(f"   Frequency: {freq} times")
            print(f"   Unique values: {len(values)}")
            if examples:
                print(f"   Examples: {examples}")

    # 스키마 생성
    schema = []
    for key in sorted(all_keys):
        values = all_values[key]
        examples = sorted(list(value_examples[key]))

        # 타입 결정
        if len(values) <= 15:  # 값이 적으면 enum
            schema_entry = {
                'name': key,
                'type': 'enum',
                'description': f'{key} 정보',
                'choices': examples[:15]
            }
        else:  # 많으면 extract
            schema_entry = {
                'name': key,
                'type': 'extract',
                'description': f'{key} 정보',
                'examples': examples[:10]
            }

        schema.append(schema_entry)

    # 스키마 저장
    with open('gold_answer_schema.json', 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Schema saved to: gold_answer_schema.json")
    print(f"   Total entity types: {len(schema)}")

    # 상세 분석 저장
    analysis = {
        'total_rows': total_rows,
        'rows_with_data': rows_with_data,
        'unique_keys': sorted(list(all_keys)),
        'key_frequency': dict(key_frequency.most_common()),
        'key_details': {
            key: {
                'frequency': key_frequency[key],
                'unique_values': len(all_values[key]),
                'top_values': all_values[key].most_common(10),
                'all_values': sorted(list(value_examples[key]))[:50]
            }
            for key in all_keys
        },
        'sample_data': sample_data
    }

    with open('gold_answer_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"✅ Analysis saved to: gold_answer_analysis.json")

    # 샘플 출력
    if sample_data:
        print("\n" + "="*60)
        print("SAMPLE DATA")
        print("="*60)

        for i, sample in enumerate(sample_data[:3], 1):
            print(f"\n[Sample {i}]")
            print(f"File: {sample['file'][:50]}")
            print(f"Text: {sample['sentence']}")
            print(f"Gold Answer Keys: {list(sample['gold_answer_nl'].keys())}")

if __name__ == "__main__":
    extract_gold_answer_nl()