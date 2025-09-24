# 119 응급 NER 스키마 분석 및 테스트 결과

## 📊 데이터 분석 결과

### CSV 파일 분석
- **경로**: `data/20250911 NER 라벨링 데이터/label/라벨링 작업 파일/all_completed_exports_20250916_163050/`
- **파일 수**: 20개 CSV 파일
- **총 레코드**: 1,574개
- **gold_answer_nl 상태**: 모두 빈 값 `{}`

### 데이터 특징
```csv
no,start,end,화자,sentence,pred_answer,gold_answer_code,gold_answer_nl
1,1.48,11.03,A,출혈은 없어요.,{},{},{}
2,11.03,17.45,A,네.,{},{},{}
```

- 119 응급 상황 대화 녹취록
- 화자: A(응급구조사), B/C(환자/보호자)
- 실제 라벨링이 되어있지 않음

## 🏗️ 생성된 스키마 (emergency_ner_schema.json)

### 14개 엔티티 타입 정의

#### 추출형 (Extract) - 9개
1. **신체부위**: 목, 배, 다리, 무릎, 손, 발 등
2. **증상**: 출혈, 통증, 골절, 의식없음, 호흡곤란 등
3. **의료행위**: 압박, 고정, CPR, 지혈 등
4. **의료장비**: 들것, 산소마스크, 붕대 등
5. **장소**: 집, 도로, 건물, 계단 등
6. **인물**: 아저씨, 어머니, 아버지 등
7. **시간**: 어제, 오늘, 방금, 10분전 등
8. **나이**: 20세, 60대, 노인, 어린이 등

#### 열거형 (Enum) - 3개
9. **의식상태**: [의식있음, 의식없음, 혼미, 반응없음, 대화가능]
10. **호흡상태**: [정상, 곤란, 없음, 빠름, 느림]
11. **긴급도**: [긴급, 준긴급, 일반, 대기가능]

#### 불린형 (Boolean) - 3개
12. **출혈여부**: true/false
13. **의식여부**: true/false
14. **호흡여부**: true/false

## 🧪 테스트 결과

### 테스트 텍스트 샘플
1. "아저씨가 목 누를 때 어때요? 괜찮아요?"
2. "이런 데 괜찮고, 배 아저씨가 누를 때 어때요?"
3. "그러면은 주들것 가져오는 게 좋을 거 같아요"

### 현재 모델 성능
- **문제점**: 5스텝만 학습한 모델이라 제대로 된 JSON 출력 불가
- **출력**: 불완전한 JSON (`{` 만 출력)

## 📝 파일 생성 목록

1. **emergency_ner_schema.json**: 119 응급 NER 스키마
2. **emergency_ner_analysis.json**: 상세 분석 결과
3. **scripts/analyze_gold_answers.py**: CSV 분석 스크립트
4. **test_emergency_ner.py**: 테스트 스크립트

## 🎯 다음 단계

### 1. 라벨링 데이터 생성
CSV의 텍스트를 기반으로 synthetic 라벨링 데이터 생성 필요:
```python
# 예시 라벨링
{
  "신체부위": ["목", "배"],
  "인물": ["아저씨"],
  "의료장비": ["들것"],
  "출혈여부": false
}
```

### 2. 학습 데이터 생성
```bash
# 응급 도메인 특화 데이터 생성
uv run python scripts/generate_structured_data.py \
  --schema emergency_ner_schema.json \
  --output ./data/emergency_ner \
  --num-samples 1000
```

### 3. 모델 학습
```bash
# 실제 학습 (3-5 에폭)
uv run python -m src.train \
  --use-lora \
  --model-type qwen3 \
  --data-path ./data/emergency_ner/train.jsonl \
  --num-epochs 3
```

### 4. 평가
```bash
# 학습된 모델로 테스트
uv run python test_emergency_ner.py
```

## 💡 인사이트

1. **라벨링 필요**: 현재 CSV의 gold_answer 필드가 비어있어 실제 라벨링 작업 필요
2. **도메인 특화**: 119 응급 상황에 특화된 엔티티 타입 정의 완료
3. **혼합 타입**: Extract, Enum, Boolean 타입 혼합하여 구조화된 출력 가능
4. **실시간 대화**: 구어체, 짧은 문장, 반복적 확인 등 특징 고려 필요

## 🔧 사용법

### 스키마 활용
```python
# Python에서 사용
import json
with open('emergency_ner_schema.json', 'r', encoding='utf-8') as f:
    schema = json.load(f)
```

### 추론 실행
```bash
# HuggingFace
uv run python inference_structured.py \
  --model-path ./outputs/model \
  --text "출혈은 없어요" \
  --schema-file emergency_ner_schema.json

# vLLM
uv run python inference_vllm_structured.py \
  --model-path ./outputs/model \
  --text "출혈은 없어요" \
  --schema-file emergency_ner_schema.json
```