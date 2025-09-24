#!/usr/bin/env python3
"""
Analyze gold_answer_nl fields from CSV files to create schema
"""

import csv
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any

def analyze_csv_files_from_list(csv_files: List) -> Dict[str, Any]:
    """Analyze CSV files from a list"""
    all_keys = set()
    all_values = defaultdict(set)
    key_frequency = Counter()
    value_examples = defaultdict(list)
    total_records = 0
    files_with_data = 0

    print(f"Processing {len(csv_files)} CSV files...")

    for csv_file in csv_files:
        print(f"Processing: {os.path.basename(csv_file)}")
        has_data = False

        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    total_records += 1

                    # Parse gold_answer_nl field
                    gold_nl = row.get('gold_answer_nl', '{}')
                    if gold_nl and gold_nl != '{}':
                        try:
                            data = json.loads(gold_nl)
                            if data:  # Not empty dict
                                has_data = True

                                for key, value in data.items():
                                    all_keys.add(key)
                                    key_frequency[key] += 1

                                    # Collect unique values
                                    if isinstance(value, list):
                                        for item in value:
                                            all_values[key].add(str(item))
                                            if len(value_examples[key]) < 10:
                                                value_examples[key].append(item)
                                    else:
                                        all_values[key].add(str(value))
                                        if len(value_examples[key]) < 10:
                                            value_examples[key].append(value)

                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            print(f"  Error processing row: {e}")

        except Exception as e:
            print(f"  Error reading file: {e}")

        if has_data:
            files_with_data += 1

    print(f"\nAnalysis Complete:")
    print(f"  Total files: {len(csv_files)}")
    print(f"  Files with data: {files_with_data}")
    print(f"  Total records: {total_records}")
    print(f"  Unique keys found: {len(all_keys)}")

    return {
        'keys': sorted(all_keys),
        'values': {k: sorted(list(v)) for k, v in all_values.items()},
        'frequency': dict(key_frequency.most_common()),
        'examples': dict(value_examples),
        'stats': {
            'total_files': len(csv_files),
            'files_with_data': files_with_data,
            'total_records': total_records,
            'unique_keys': len(all_keys)
        }
    }

def analyze_csv_files(directory: str) -> Dict[str, Any]:
    """Analyze all CSV files in directory"""
    all_keys = set()
    all_values = defaultdict(set)
    key_frequency = Counter()
    value_examples = defaultdict(list)
    total_records = 0
    files_with_data = 0

    # Find all CSV files
    csv_files = list(Path(directory).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        has_data = False

        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    total_records += 1

                    # Parse gold_answer_nl field
                    gold_nl = row.get('gold_answer_nl', '{}')
                    if gold_nl and gold_nl != '{}':
                        try:
                            data = json.loads(gold_nl)
                            if data:  # Not empty dict
                                has_data = True

                                for key, value in data.items():
                                    all_keys.add(key)
                                    key_frequency[key] += 1

                                    # Collect unique values
                                    if isinstance(value, list):
                                        for item in value:
                                            all_values[key].add(str(item))
                                            if len(value_examples[key]) < 10:
                                                value_examples[key].append(item)
                                    else:
                                        all_values[key].add(str(value))
                                        if len(value_examples[key]) < 10:
                                            value_examples[key].append(value)

                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            print(f"  Error processing row: {e}")

        except Exception as e:
            print(f"  Error reading file: {e}")

        if has_data:
            files_with_data += 1

    print(f"\nAnalysis Complete:")
    print(f"  Total files: {len(csv_files)}")
    print(f"  Files with data: {files_with_data}")
    print(f"  Total records: {total_records}")
    print(f"  Unique keys found: {len(all_keys)}")

    return {
        'keys': sorted(all_keys),
        'values': {k: sorted(list(v)) for k, v in all_values.items()},
        'frequency': dict(key_frequency.most_common()),
        'examples': dict(value_examples),
        'stats': {
            'total_files': len(csv_files),
            'files_with_data': files_with_data,
            'total_records': total_records,
            'unique_keys': len(all_keys)
        }
    }

def create_schema(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create schema based on analysis"""
    schema = []

    # Define known entity types based on frequency and examples
    entity_mappings = {
        # Medical/Emergency entities
        '신체부위': {
            'type': 'extract',
            'description': '신체 부위 (목, 배, 다리, 팔 등)'
        },
        '증상': {
            'type': 'extract',
            'description': '의료 증상 (출혈, 통증, 골절 등)'
        },
        '의료장비': {
            'type': 'extract',
            'description': '의료 장비 및 도구'
        },

        # Location/Place
        '장소': {
            'type': 'extract',
            'description': '장소 및 위치'
        },
        '주소': {
            'type': 'extract',
            'description': '구체적인 주소'
        },

        # Person/Organization
        '사람': {
            'type': 'extract',
            'description': '사람 이름 및 호칭'
        },
        '조직': {
            'type': 'extract',
            'description': '조직 및 기관명'
        },

        # Time/Number
        '시간': {
            'type': 'extract',
            'description': '시간 정보'
        },
        '수량': {
            'type': 'extract',
            'description': '수량 및 숫자 정보'
        },

        # Status
        '긴급도': {
            'type': 'enum',
            'description': '긴급 정도',
            'choices': ['긴급', '준긴급', '일반']
        },
        '의식상태': {
            'type': 'enum',
            'description': '환자 의식 상태',
            'choices': ['의식있음', '의식없음', '혼미']
        }
    }

    # Add found keys to schema
    for key in analysis['keys']:
        if key in entity_mappings:
            schema_entry = {
                'name': key,
                **entity_mappings[key]
            }

            # Add examples if available
            if key in analysis['examples']:
                examples = analysis['examples'][key][:5]  # First 5 examples
                schema_entry['examples'] = examples

            schema.append(schema_entry)
        else:
            # For unknown keys, create extract type
            schema.append({
                'name': key,
                'type': 'extract',
                'description': f'{key} 정보',
                'examples': analysis['examples'].get(key, [])[:5]
            })

    # If no keys found, add default schema
    if not schema:
        print("No entities found in data, creating default schema")
        schema = [
            {
                'name': '신체부위',
                'type': 'extract',
                'description': '신체 부위 (목, 배, 다리, 팔 등)'
            },
            {
                'name': '증상',
                'type': 'extract',
                'description': '의료 증상 (출혈, 통증, 골절 등)'
            },
            {
                'name': '장소',
                'type': 'extract',
                'description': '장소 및 위치'
            },
            {
                'name': '긴급도',
                'type': 'enum',
                'description': '긴급 정도',
                'choices': ['긴급', '준긴급', '일반']
            }
        ]

    return schema

def main():
    # Directory with CSV files - using glob to find it
    import glob
    csv_patterns = [
        "./data/**/all_completed_exports_20250916_163050/*.csv",
        "./data/**/*.csv"
    ]

    csv_files = []
    for pattern in csv_patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            csv_files = files
            csv_dir = os.path.dirname(files[0])
            break

    if not csv_files:
        print("No CSV files found!")
        return

    print(f"Found {len(csv_files)} CSV files in: {csv_dir}")
    print("="*60)

    # Analyze files directly
    analysis = analyze_csv_files_from_list(csv_files)

    # Print analysis results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    if analysis['keys']:
        print("\nFound Entity Types:")
        for key, freq in analysis['frequency'].items():
            print(f"  - {key}: {freq} occurrences")
            if key in analysis['examples'] and analysis['examples'][key]:
                examples = analysis['examples'][key][:3]
                print(f"    Examples: {examples}")
    else:
        print("\nNo entities found in gold_answer_nl fields")
        print("All fields appear to be empty {}")

    # Create schema
    schema = create_schema(analysis)

    # Save schema
    schema_file = "emergency_ner_schema.json"
    with open(schema_file, 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Schema saved to: {schema_file}")
    print(f"   Total entity types: {len(schema)}")

    # Save detailed analysis
    analysis_file = "emergency_ner_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"✅ Detailed analysis saved to: {analysis_file}")

if __name__ == "__main__":
    main()