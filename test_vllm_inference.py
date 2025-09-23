#!/usr/bin/env python3
"""
Comprehensive test suite for vLLM structured inference
Tests the vLLM GuidedDecodingParams API with various scenarios
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    print("Please install vLLM first: pip install vllm")
    sys.exit(1)


class TestStatus(Enum):
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    SKIPPED = "⏭️ SKIPPED"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str = ""
    output: Any = None


class VLLMStructuredTester:
    """Test suite for vLLM structured inference"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        """Initialize test environment"""
        self.model_name = model_name
        self.llm = None
        self.results: List[TestResult] = []

    def setup(self):
        """Setup vLLM model"""
        try:
            print(f"Loading model: {self.model_name}")
            self.llm = LLM(
                model=self.model_name,
                max_model_len=512,
                gpu_memory_utilization=0.5
            )
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def test_basic_json_schema(self):
        """Test 1: Basic JSON schema with simple fields"""
        test_name = "Basic JSON Schema"
        print(f"\n{'='*50}")
        print(f"Test 1: {test_name}")
        print('='*50)

        try:
            # Define simple schema
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "active": {"type": "boolean"}
                },
                "required": ["name", "age", "active"]
            }

            prompt = "Generate a person profile: John is 30 years old and currently active."

            guided_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                guided_decoding=guided_params,
                temperature=0.1,
                max_tokens=50
            )

            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            result = outputs[0].outputs[0].text

            # Parse and validate
            parsed = json.loads(result)
            assert "name" in parsed
            assert "age" in parsed
            assert "active" in parsed

            print(f"Output: {json.dumps(parsed, indent=2)}")
            self.results.append(TestResult(test_name, TestStatus.PASSED, output=parsed))

        except Exception as e:
            print(f"Error: {e}")
            self.results.append(TestResult(test_name, TestStatus.FAILED, str(e)))

    def test_array_extraction(self):
        """Test 2: Array extraction (EXTRACT type fields)"""
        test_name = "Array Extraction"
        print(f"\n{'='*50}")
        print(f"Test 2: {test_name}")
        print('='*50)

        try:
            schema = {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "organizations": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["people", "organizations"]
            }

            prompt = ("Extract entities from: Tim Cook from Apple met with "
                     "Satya Nadella from Microsoft to discuss AI partnership.")

            guided_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                guided_decoding=guided_params,
                temperature=0.1,
                max_tokens=100
            )

            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            result = outputs[0].outputs[0].text
            parsed = json.loads(result)

            print(f"Output: {json.dumps(parsed, indent=2)}")
            assert isinstance(parsed["people"], list)
            assert isinstance(parsed["organizations"], list)

            self.results.append(TestResult(test_name, TestStatus.PASSED, output=parsed))

        except Exception as e:
            print(f"Error: {e}")
            self.results.append(TestResult(test_name, TestStatus.FAILED, str(e)))

    def test_enum_constraints(self):
        """Test 3: Enum constraints (ENUM type fields)"""
        test_name = "Enum Constraints"
        print(f"\n{'='*50}")
        print(f"Test 3: {test_name}")
        print('='*50)

        try:
            schema = {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"]
                    },
                    "category": {
                        "type": "string",
                        "enum": ["business", "technology", "politics", "sports"]
                    }
                },
                "required": ["sentiment", "category"]
            }

            prompt = "Classify: Apple announces record profits with new AI features."

            guided_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                guided_decoding=guided_params,
                temperature=0.1,
                max_tokens=50
            )

            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            result = outputs[0].outputs[0].text
            parsed = json.loads(result)

            print(f"Output: {json.dumps(parsed, indent=2)}")
            assert parsed["sentiment"] in ["positive", "negative", "neutral"]
            assert parsed["category"] in ["business", "technology", "politics", "sports"]

            self.results.append(TestResult(test_name, TestStatus.PASSED, output=parsed))

        except Exception as e:
            print(f"Error: {e}")
            self.results.append(TestResult(test_name, TestStatus.FAILED, str(e)))

    def test_mixed_schema(self):
        """Test 4: Mixed schema (like test_schema.json)"""
        test_name = "Mixed Schema"
        print(f"\n{'='*50}")
        print(f"Test 4: {test_name}")
        print('='*50)

        try:
            # Load test_schema.json style
            schema = {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of people mentioned"
                    },
                    "organizations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Organizations or companies mentioned"
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                        "description": "Overall sentiment"
                    },
                    "contains_financial_info": {
                        "type": "boolean",
                        "description": "Whether text contains financial information"
                    }
                },
                "required": ["people", "organizations", "sentiment", "contains_financial_info"]
            }

            prompt = ("Analyze: Tim Cook announced Apple's quarterly earnings "
                     "of $123 billion, exceeding analyst expectations.")

            guided_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                guided_decoding=guided_params,
                temperature=0.1,
                max_tokens=150
            )

            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            result = outputs[0].outputs[0].text
            parsed = json.loads(result)

            print(f"Output: {json.dumps(parsed, indent=2)}")

            # Validate structure
            assert isinstance(parsed["people"], list)
            assert isinstance(parsed["organizations"], list)
            assert parsed["sentiment"] in ["positive", "negative", "neutral"]
            assert isinstance(parsed["contains_financial_info"], bool)

            self.results.append(TestResult(test_name, TestStatus.PASSED, output=parsed))

        except Exception as e:
            print(f"Error: {e}")
            self.results.append(TestResult(test_name, TestStatus.FAILED, str(e)))

    def test_batch_processing(self):
        """Test 5: Batch processing with multiple prompts"""
        test_name = "Batch Processing"
        print(f"\n{'='*50}")
        print(f"Test 5: {test_name}")
        print('='*50)

        try:
            schema = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["summary", "keywords"]
            }

            prompts = [
                "Summarize: AI technology is advancing rapidly.",
                "Summarize: Climate change requires immediate action.",
                "Summarize: Stock markets showed mixed results today."
            ]

            guided_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                guided_decoding=guided_params,
                temperature=0.1,
                max_tokens=100
            )

            outputs = self.llm.generate(prompts, sampling_params=sampling_params)

            results = []
            for i, output in enumerate(outputs):
                result = output.outputs[0].text
                parsed = json.loads(result)
                results.append(parsed)
                print(f"Prompt {i+1}: {json.dumps(parsed, indent=2)}")

            assert len(results) == len(prompts)
            self.results.append(TestResult(test_name, TestStatus.PASSED, output=results))

        except Exception as e:
            print(f"Error: {e}")
            self.results.append(TestResult(test_name, TestStatus.FAILED, str(e)))

    def test_no_schema(self):
        """Test 6: Generation without schema (baseline)"""
        test_name = "No Schema (Baseline)"
        print(f"\n{'='*50}")
        print(f"Test 6: {test_name}")
        print('='*50)

        try:
            prompt = "Generate a JSON with name and age for a person named Alice who is 25."

            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=50
            )

            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            result = outputs[0].outputs[0].text

            print(f"Output (raw): {result}")

            # Try to parse as JSON
            try:
                parsed = json.loads(result)
                print(f"Parsed successfully: {parsed}")
                self.results.append(TestResult(test_name, TestStatus.PASSED, output=parsed))
            except json.JSONDecodeError:
                print("Note: Without schema, output may not be valid JSON")
                self.results.append(TestResult(test_name, TestStatus.PASSED,
                                              "Output generated but not valid JSON", result))

        except Exception as e:
            print(f"Error: {e}")
            self.results.append(TestResult(test_name, TestStatus.FAILED, str(e)))

    def test_korean_text(self):
        """Test 7: Korean text processing"""
        test_name = "Korean Text"
        print(f"\n{'='*50}")
        print(f"Test 7: {test_name}")
        print('='*50)

        try:
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

            prompt = "Extract from: 김철수 대표가 삼성전자와 100억원 규모의 계약을 체결했습니다."

            guided_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(
                guided_decoding=guided_params,
                temperature=0.1,
                max_tokens=100
            )

            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            result = outputs[0].outputs[0].text
            parsed = json.loads(result)

            print(f"Output: {json.dumps(parsed, indent=2, ensure_ascii=False)}")

            self.results.append(TestResult(test_name, TestStatus.PASSED, output=parsed))

        except Exception as e:
            print(f"Error: {e}")
            self.results.append(TestResult(test_name, TestStatus.FAILED, str(e)))

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("vLLM Structured Output Test Suite")
        print("="*60)

        if not self.setup():
            print("Failed to setup vLLM. Exiting.")
            return False

        # Run tests
        self.test_basic_json_schema()
        self.test_array_extraction()
        self.test_enum_constraints()
        self.test_mixed_schema()
        self.test_batch_processing()
        self.test_no_schema()
        self.test_korean_text()

        # Summary
        self.print_summary()

        # Return True if all tests passed
        return all(r.status == TestStatus.PASSED for r in self.results)

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        for result in self.results:
            print(f"{result.status.value} {result.name}")
            if result.message:
                print(f"    {result.message}")

        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

        print("\n" + "-"*60)
        print(f"Total: {len(self.results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")

        if failed == 0:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️ {failed} test(s) failed")


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM structured output")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to test with"
    )
    parser.add_argument(
        "--custom-model",
        type=str,
        help="Path to custom trained model"
    )

    args = parser.parse_args()

    # Use custom model if provided
    model = args.custom_model if args.custom_model else args.model

    # Run tests
    tester = VLLMStructuredTester(model)
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()