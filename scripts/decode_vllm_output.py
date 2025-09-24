#!/usr/bin/env python3
"""
Decode and display vLLM output with proper Korean text
"""

import json
import sys

def decode_unicode_escapes(text):
    """Decode Unicode escape sequences to actual characters"""
    try:
        # Decode Unicode escapes
        decoded = text.encode('utf-8').decode('unicode_escape')
        # Fix encoding issues for Korean text
        decoded = decoded.encode('latin-1').decode('utf-8')
        return decoded
    except:
        return text

def fix_truncated_json(text):
    """Try to fix truncated JSON by adding missing brackets"""
    # Count brackets
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    # Add missing closing characters
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)

    return text

def main():
    # Read the vLLM output file
    with open('test_korean_result.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        print("=" * 60)
        print(f"Input: {item['input']}")
        print("=" * 60)

        output = item['output']
        if 'raw_output' in output:
            raw = output['raw_output']

            # Decode Unicode escapes
            decoded = decode_unicode_escapes(raw)

            # Try to fix and parse JSON
            fixed = fix_truncated_json(decoded)

            try:
                parsed = json.loads(fixed)
                print("Successfully parsed output:")
                print(json.dumps(parsed, ensure_ascii=False, indent=2))

                # Show some key values
                print("\n주요 추출 정보:")
                if "환자 증상1" in parsed:
                    print(f"  환자 증상1: {parsed['환자 증상1']}")
                if "의식상태_1차" in parsed:
                    print(f"  의식상태_1차: {parsed['의식상태_1차']}")
                if "활력징후_1차_맥박" in parsed:
                    print(f"  활력징후_1차_맥박: {parsed['활력징후_1차_맥박']}")
                if "활력징후_1차_체온" in parsed:
                    print(f"  활력징후_1차_체온: {parsed['활력징후_1차_체온']}")

            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print("\nDecoded text (first 1000 chars):")
                print(decoded[:1000])
        else:
            # Already parsed successfully
            print("Output:")
            print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()