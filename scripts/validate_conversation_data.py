#!/usr/bin/env python3
"""
Validate conversation format data for NER training
Ensures data quality and correct format
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))


class ConversationValidator:
    """Validate conversation format data for quality and correctness"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues = []
        self.stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "empty_responses": 0,
            "malformed_json": 0,
            "missing_entities": 0,
            "entity_types": Counter(),
            "message_lengths": [],
            "turn_counts": []
        }

    def validate_message_structure(self, message: Dict) -> Tuple[bool, str]:
        """Validate a single message structure"""
        if not isinstance(message, dict):
            return False, "Message must be a dictionary"

        if "role" not in message:
            return False, "Message missing 'role' field"

        if "content" not in message:
            return False, "Message missing 'content' field"

        if message["role"] not in ["user", "assistant", "system"]:
            return False, f"Invalid role: {message['role']}"

        if not isinstance(message["content"], str):
            return False, "Message content must be a string"

        if len(message["content"].strip()) == 0:
            return False, "Message content is empty"

        return True, ""

    def validate_conversation(self, conv_data: Dict, index: int) -> bool:
        """Validate a single conversation entry"""
        issues_found = []

        # Check basic structure
        if not isinstance(conv_data, dict):
            issues_found.append(f"Sample {index}: Not a dictionary")
            self.stats["invalid_samples"] += 1
            return False

        if "messages" not in conv_data:
            issues_found.append(f"Sample {index}: Missing 'messages' field")
            self.stats["invalid_samples"] += 1
            return False

        messages = conv_data["messages"]

        if not isinstance(messages, list):
            issues_found.append(f"Sample {index}: 'messages' must be a list")
            self.stats["invalid_samples"] += 1
            return False

        if len(messages) < 2:
            issues_found.append(f"Sample {index}: Conversation must have at least 2 messages")
            self.stats["invalid_samples"] += 1
            return False

        # Validate each message
        for msg_idx, message in enumerate(messages):
            valid, error = self.validate_message_structure(message)
            if not valid:
                issues_found.append(f"Sample {index}, Message {msg_idx}: {error}")

        # Check conversation flow
        if messages[0]["role"] != "user":
            issues_found.append(f"Sample {index}: First message should be from 'user'")

        # Check alternating roles (user -> assistant -> user -> assistant)
        for i in range(len(messages) - 1):
            if messages[i]["role"] == messages[i + 1]["role"]:
                issues_found.append(
                    f"Sample {index}: Consecutive messages from same role at positions {i} and {i+1}"
                )

        # Validate assistant responses
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        if not assistant_messages:
            issues_found.append(f"Sample {index}: No assistant responses found")
            self.stats["invalid_samples"] += 1
            return False

        # Check for NER output in assistant responses
        for msg_idx, assistant_msg in enumerate(assistant_messages):
            content = assistant_msg["content"].strip()

            # Check if it looks like JSON
            if content.startswith("{") and content.endswith("}"):
                try:
                    entities = json.loads(content)

                    # Validate entity structure
                    if not isinstance(entities, dict):
                        issues_found.append(
                            f"Sample {index}: Assistant response {msg_idx} is not a valid entity dict"
                        )
                    else:
                        # Count entity types
                        for entity_type, entity_list in entities.items():
                            if not isinstance(entity_list, list):
                                issues_found.append(
                                    f"Sample {index}: Entity type '{entity_type}' value is not a list"
                                )
                            else:
                                self.stats["entity_types"][entity_type] += len(entity_list)

                        # Check for empty entity response
                        if not entities:
                            self.stats["empty_responses"] += 1

                except json.JSONDecodeError:
                    issues_found.append(
                        f"Sample {index}: Assistant response {msg_idx} contains invalid JSON"
                    )
                    self.stats["malformed_json"] += 1
            else:
                # Non-JSON response (might be follow-up or explanation)
                if msg_idx == 0:  # First assistant response should typically be JSON
                    if self.verbose:
                        issues_found.append(
                            f"Sample {index}: First assistant response is not JSON (might be intentional)"
                        )

        # Update statistics
        self.stats["turn_counts"].append(len(messages))
        self.stats["message_lengths"].extend([len(m["content"]) for m in messages])

        # Record issues if any
        if issues_found:
            self.issues.extend(issues_found)
            self.stats["invalid_samples"] += 1
            return False
        else:
            self.stats["valid_samples"] += 1
            return True

    def validate_file(self, file_path: Path) -> bool:
        """Validate an entire JSONL file"""
        print(f"Validating {file_path}...")

        all_valid = True
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                self.stats["total_samples"] += 1

                try:
                    data = json.loads(line)
                    if not self.validate_conversation(data, line_num):
                        all_valid = False

                except json.JSONDecodeError as e:
                    self.issues.append(f"Line {line_num}: JSON decode error - {e}")
                    self.stats["invalid_samples"] += 1
                    all_valid = False

        return all_valid

    def print_report(self):
        """Print validation report"""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)

        # Basic stats
        print(f"\nBasic Statistics:")
        print(f"  Total samples: {self.stats['total_samples']}")
        print(f"  Valid samples: {self.stats['valid_samples']} "
              f"({100 * self.stats['valid_samples'] / max(1, self.stats['total_samples']):.1f}%)")
        print(f"  Invalid samples: {self.stats['invalid_samples']} "
              f"({100 * self.stats['invalid_samples'] / max(1, self.stats['total_samples']):.1f}%)")

        # Conversation statistics
        if self.stats["turn_counts"]:
            print(f"\nConversation Statistics:")
            print(f"  Total conversations: {len(self.stats['turn_counts'])}")
            avg_turns = sum(self.stats["turn_counts"]) / len(self.stats["turn_counts"])
            print(f"  Average messages per conversation: {avg_turns:.1f}")
            if max(self.stats['turn_counts']) > 2:
                print(f"  Note: Found conversations with {max(self.stats['turn_counts'])} messages (expected 2 for single-turn)")

        # Entity statistics
        print(f"\nEntity Statistics:")
        if self.stats["entity_types"]:
            print("  Entity type distribution:")
            for entity_type, count in self.stats["entity_types"].most_common(10):
                print(f"    {entity_type}: {count}")
        print(f"  Empty responses: {self.stats['empty_responses']}")
        print(f"  Malformed JSON: {self.stats['malformed_json']}")

        # Message length stats
        if self.stats["message_lengths"]:
            avg_length = sum(self.stats["message_lengths"]) / len(self.stats["message_lengths"])
            print(f"\nMessage Statistics:")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Min length: {min(self.stats['message_lengths'])} characters")
            print(f"  Max length: {max(self.stats['message_lengths'])} characters")

        # Issues summary
        if self.issues:
            print(f"\n⚠️  Issues Found ({len(self.issues)}):")
            # Show first 10 issues
            for issue in self.issues[:10]:
                print(f"  • {issue}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more issues")
        else:
            print(f"\n✅ No issues found!")

        # Overall result
        print("\n" + "="*60)
        if self.stats["invalid_samples"] == 0:
            print("✅ VALIDATION PASSED - All samples are valid!")
        elif self.stats["invalid_samples"] / self.stats["total_samples"] < 0.05:
            print("⚠️  VALIDATION PASSED WITH WARNINGS - Less than 5% invalid samples")
        else:
            print("❌ VALIDATION FAILED - Too many invalid samples")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Validate conversation format NER data")
    parser.add_argument("input", type=str, help="Input JSONL file or directory to validate")
    parser.add_argument("--verbose", action="store_true",
                       help="Show all validation issues")
    parser.add_argument("--strict", action="store_true",
                       help="Strict validation mode")
    parser.add_argument("--fix", action="store_true",
                       help="Attempt to fix common issues")
    parser.add_argument("--output", type=str,
                       help="Output file for fixed data (requires --fix)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    validator = ConversationValidator(verbose=args.verbose)

    if input_path.is_dir():
        # Validate all JSONL files in directory
        print(f"Validating all JSONL files in {input_path}")
        all_valid = True

        for jsonl_file in sorted(input_path.glob("*.jsonl")):
            print(f"\n{'='*60}")
            print(f"File: {jsonl_file.name}")
            print('='*60)

            file_validator = ConversationValidator(verbose=args.verbose)
            file_valid = file_validator.validate_file(jsonl_file)
            file_validator.print_report()

            if not file_valid:
                all_valid = False

        print("\n" + "="*60)
        print("OVERALL VALIDATION RESULT")
        print("="*60)
        if all_valid:
            print("✅ All files passed validation!")
        else:
            print("❌ Some files failed validation")

    else:
        # Validate single file
        valid = validator.validate_file(input_path)
        validator.print_report()

        if args.fix and not valid:
            print("\nAttempting to fix issues...")
            # TODO: Implement auto-fix functionality
            print("Auto-fix not yet implemented")


if __name__ == "__main__":
    main()