#!/usr/bin/env python3
"""
실제 CSV 파일에서 엔티티를 기계적으로 추출
"""

import csv
import json
import os
import glob
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any
import re

def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """텍스트에서 패턴 기반으로 엔티티 추출"""
    entities = defaultdict(list)

    # 신체 부위 패턴
    body_parts = re.findall(r'(목|머리|배|다리|무릎|발|손|팔|가슴|등|허리|어깨|코|눈|귀|입)', text)
    if body_parts:
        entities['신체부위'] = list(set(body_parts))

    # 증상 패턴
    symptoms = re.findall(r'(출혈|통증|아프|아픈|골절|부러|의식|호흡|숨|토하|구토|어지럽|열|고열)', text)
    if symptoms:
        entities['증상'] = list(set(symptoms))

    # 의료 장비
    equipment = re.findall(r'(들것|주사|약|붕대|거즈|산소|마스크|목보호대)', text)
    if equipment:
        entities['의료장비'] = list(set(equipment))

    # 장소
    places = re.findall(r'(집|화장실|거실|방|계단|도로|길|건물|병원|학교|회사)', text)
    if places:
        entities['장소'] = list(set(places))

    # 시간 표현
    times = re.findall(r'(어제|오늘|내일|아까|방금|지금|새벽|아침|점심|저녁|밤|\d+시|\d+분)', text)
    if times:
        entities['시간'] = list(set(times))

    # 나이
    ages = re.findall(r'(\d+세|\d+살|어린이|아이|청소년|어른|노인|할머니|할아버지)', text)
    if ages:
        entities['나이'] = list(set(ages))

    # 인물/호칭
    people = re.findall(r'(아저씨|아줌마|어머니|아버지|엄마|아빠|형|누나|동생|친구)', text)
    if people:
        entities['인물'] = list(set(people))

    # 숫자/수량
    numbers = re.findall(r'(\d+번|\d+개|\d+명|\d+층|\d+미터|\d+cm)', text)
    if numbers:
        entities['수량'] = list(set(numbers))

    # 행위/동작
    actions = re.findall(r'(누르|누를|돌리|펴|굽히|움직이|걷|앉|서|눕)', text)
    if actions:
        entities['행위'] = list(set(actions))

    # 상태 표현
    states = re.findall(r'(괜찮|아프|불편|어떠|어때|좋|나쁘|심하)', text)
    if states:
        entities['상태'] = list(set(states))

    return dict(entities)

def analyze_all_csv_files():
    """모든 CSV 파일 분석하여 엔티티 추출"""

    # CSV 파일 찾기
    csv_files = glob.glob("./data/**/*.csv", recursive=True)[:20]

    if not csv_files:
        print("No CSV files found!")
        return

    print(f"Found {len(csv_files)} CSV files")

    # 전체 통계
    all_entities = defaultdict(Counter)
    entity_examples = defaultdict(set)
    total_sentences = 0
    sentences_with_entities = 0

    # 각 CSV 파일 처리
    for csv_file in csv_files:
        print(f"\nProcessing: {os.path.basename(csv_file)}")

        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    sentence = row.get('sentence', '').strip()

                    if sentence:
                        total_sentences += 1

                        # 엔티티 추출
                        entities = extract_entities_from_text(sentence)

                        if entities:
                            sentences_with_entities += 1

                            # 통계 업데이트
                            for entity_type, values in entities.items():
                                for value in values:
                                    all_entities[entity_type][value] += 1
                                    entity_examples[entity_type].add(value)

                        # gold_answer_nl 확인
                        gold_nl = row.get('gold_answer_nl', '{}')
                        if gold_nl and gold_nl != '{}':
                            try:
                                gold_data = json.loads(gold_nl)
                                if gold_data:
                                    print(f"  Found labeled data: {gold_data}")
                                    # 라벨된 데이터도 통계에 추가
                                    for key, values in gold_data.items():
                                        if isinstance(values, list):
                                            for value in values:
                                                all_entities[key][value] += 1
                                                entity_examples[key].add(value)
                                        else:
                                            all_entities[key][values] += 1
                                            entity_examples[key].add(values)
                            except:
                                pass

        except Exception as e:
            print(f"  Error: {e}")

    # 결과 출력
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    print(f"Total sentences: {total_sentences}")
    print(f"Sentences with entities: {sentences_with_entities}")
    print(f"Coverage: {sentences_with_entities/total_sentences*100:.1f}%")

    print("\n" + "="*60)
    print("ENTITY TYPES FOUND")
    print("="*60)

    for entity_type in sorted(all_entities.keys()):
        values = all_entities[entity_type]
        examples = sorted(entity_examples[entity_type])[:10]

        print(f"\n{entity_type} ({len(values)} unique values)")
        print(f"  Most common: {values.most_common(5)}")
        print(f"  Examples: {examples[:5]}")

    # 스키마 생성
    schema = create_schema_from_entities(all_entities, entity_examples)

    # 스키마 저장
    with open('extracted_entities_schema.json', 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Schema saved to: extracted_entities_schema.json")

    # 상세 결과 저장
    analysis_result = {
        'total_sentences': total_sentences,
        'sentences_with_entities': sentences_with_entities,
        'entity_types': {
            entity_type: {
                'count': len(values),
                'top_10': values.most_common(10),
                'examples': sorted(list(entity_examples[entity_type]))[:20]
            }
            for entity_type, values in all_entities.items()
        }
    }

    with open('extracted_entities_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    print(f"✅ Analysis saved to: extracted_entities_analysis.json")

def create_schema_from_entities(all_entities, entity_examples):
    """추출된 엔티티로부터 스키마 생성"""
    schema = []

    for entity_type in sorted(all_entities.keys()):
        values = all_entities[entity_type]
        examples = sorted(list(entity_examples[entity_type]))[:10]

        # 값이 제한적이면 enum, 아니면 extract
        if len(values) <= 10:
            schema_entry = {
                'name': entity_type,
                'type': 'enum',
                'description': f'{entity_type} 정보',
                'choices': sorted(list(entity_examples[entity_type]))
            }
        else:
            schema_entry = {
                'name': entity_type,
                'type': 'extract',
                'description': f'{entity_type} 정보',
                'examples': examples[:5]
            }

        schema.append(schema_entry)

    return schema

if __name__ == "__main__":
    print("Extracting entities from CSV files...")
    print("="*60)
    analyze_all_csv_files()