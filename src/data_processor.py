import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd


class NERDataProcessor:
    def __init__(self,
                 max_length: int = 512,
                 instruction_template: Optional[str] = None):
        self.max_length = max_length
        self.instruction_template = instruction_template or "Extract named entities from the following text and return them in JSON format."

    def format_for_training(self, text: str, entities: Dict[str, List[str]],
                           add_instruction: bool = True) -> Dict[str, str]:
        prompt = ""
        if add_instruction:
            prompt = f"{self.instruction_template}\n\nText: {text}\n\nEntities:"
        else:
            prompt = f"Text: {text}\n\nEntities:"

        response = json.dumps(entities, ensure_ascii=False, indent=2)

        return {
            "instruction": self.instruction_template if add_instruction else "",
            "input": text,
            "output": response,
            "text": f"{prompt} {response}"  # For SFTTrainer
        }

    def load_jsonl(self, file_path: Union[str, Path]) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def save_jsonl(self, data: List[Dict], file_path: Union[str, Path]):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def process_conll_format(self, text: str) -> Dict[str, List[str]]:
        entities = {}
        current_entity = []
        current_label = None

        for line in text.strip().split('\n'):
            if not line.strip():
                if current_entity and current_label:
                    label_type = current_label.replace('B-', '').replace('I-', '')
                    if label_type not in entities:
                        entities[label_type] = []
                    entities[label_type].append(' '.join(current_entity))
                    current_entity = []
                    current_label = None
                continue

            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                label = parts[-1]

                if label.startswith('B-'):
                    if current_entity and current_label:
                        label_type = current_label.replace('B-', '').replace('I-', '')
                        if label_type not in entities:
                            entities[label_type] = []
                        entities[label_type].append(' '.join(current_entity))
                    current_entity = [token]
                    current_label = label
                elif label.startswith('I-') and current_label:
                    current_entity.append(token)
                elif label == 'O':
                    if current_entity and current_label:
                        label_type = current_label.replace('B-', '').replace('I-', '')
                        if label_type not in entities:
                            entities[label_type] = []
                        entities[label_type].append(' '.join(current_entity))
                        current_entity = []
                        current_label = None

        if current_entity and current_label:
            label_type = current_label.replace('B-', '').replace('I-', '')
            if label_type not in entities:
                entities[label_type] = []
            entities[label_type].append(' '.join(current_entity))

        return entities

    def create_dataset_from_jsonl(self,
                                  train_path: Optional[str] = None,
                                  val_path: Optional[str] = None,
                                  test_path: Optional[str] = None) -> DatasetDict:
        dataset_dict = {}

        if train_path:
            train_data = self.load_jsonl(train_path)
            dataset_dict['train'] = Dataset.from_list(train_data)

        if val_path:
            val_data = self.load_jsonl(val_path)
            dataset_dict['validation'] = Dataset.from_list(val_data)

        if test_path:
            test_data = self.load_jsonl(test_path)
            dataset_dict['test'] = Dataset.from_list(test_data)

        return DatasetDict(dataset_dict)

    def prepare_for_sft(self, examples: Dict) -> Dict:
        texts = []
        for i in range(len(examples['input'])):
            instruction = examples.get('instruction', [''] * len(examples['input']))[i]
            input_text = examples['input'][i]
            output = examples['output'][i]

            if instruction:
                prompt = f"{instruction}\n\nText: {input_text}\n\nEntities: {output}"
            else:
                prompt = f"Text: {input_text}\n\nEntities: {output}"

            texts.append(prompt)

        return {'text': texts}

    def load_opensource_dataset(self, dataset_name: str = "conll2003") -> DatasetDict:
        if dataset_name == "conll2003":
            dataset = load_dataset("conll2003")

            processed_dataset = {}
            for split in dataset.keys():
                examples = []
                for item in dataset[split]:
                    tokens = item['tokens']
                    ner_tags = item['ner_tags']

                    text = ' '.join(tokens)
                    entities = self._extract_entities_from_tags(tokens, ner_tags, dataset_name)

                    formatted = self.format_for_training(text, entities)
                    examples.append(formatted)

                processed_dataset[split] = Dataset.from_list(examples)

            return DatasetDict(processed_dataset)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported yet")

    def _extract_entities_from_tags(self, tokens: List[str], tags: List[int],
                                   dataset_name: str) -> Dict[str, List[str]]:
        if dataset_name == "conll2003":
            tag_names = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC',
                        'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        entities = {}
        current_entity = []
        current_type = None

        for token, tag_idx in zip(tokens, tags):
            tag = tag_names[tag_idx] if tag_idx < len(tag_names) else 'O'

            if tag.startswith('B-'):
                if current_entity and current_type:
                    if current_type not in entities:
                        entities[current_type] = []
                    entities[current_type].append(' '.join(current_entity))
                current_entity = [token]
                current_type = tag[2:]
            elif tag.startswith('I-') and current_type == tag[2:]:
                current_entity.append(token)
            else:
                if current_entity and current_type:
                    if current_type not in entities:
                        entities[current_type] = []
                    entities[current_type].append(' '.join(current_entity))
                current_entity = []
                current_type = None

        if current_entity and current_type:
            if current_type not in entities:
                entities[current_type] = []
            entities[current_type].append(' '.join(current_entity))

        return entities