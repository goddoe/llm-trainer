# NER 라벨링 데이터 참고 문서

## 데이터셋 개요

### 파일 정보
- **데이터 경로**: `data/20250911 NER 라벨링 데이터/label/라벨링 작업 파일/`
- **파일 형식**: CSV (UTF-8 with BOM)
- **파일명 패턴**: `{datetime}_{system}_{timestamp}_nlp_export_{export_timestamp}.csv`
  - 예시: `202105061100012_ai-stt-relay002_20250604_095031_nlp_export_20250916_163050.csv`

### 데이터 구조

CSV 파일은 다음과 같은 컬럼을 포함합니다:

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| `no` | 문장 번호 | 1, 2, 3... |
| `start` | 발화 시작 시간 (초) | 1.48 |
| `end` | 발화 종료 시간 (초) | 11.03 |
| `화자` | 발화자 구분 | A, B, C |
| `sentence` | 실제 발화 텍스트 | "출혈은 없어요." |
| `pred_answer` | 예측된 NER 결과 (JSON) | {} |
| `gold_answer_code` | 정답 NER 코드 (JSON) | {} |
| `gold_answer_nl` | 정답 NER 자연어 (JSON) | {} |

## 데이터 특징

### 1. 도메인
- **119 응급 상황 대화**: 응급 구조사와 환자/보호자 간의 대화
- **의료 용어**: 출혈, 통증, 신체 부위 등 의료 관련 용어 포함
- **실시간 대화**: 실제 응급 상황에서의 자연스러운 구어체

### 2. 화자 구분
- **A**: 주로 응급 구조사
- **B**: 환자 또는 보호자
- **C**: 추가 구조사 또는 관계자

### 3. 텍스트 특성
```
예시 발화:
- "출혈은 없어요."
- "아저씨가 목 누를 때 어때요? 괜찮아요?"
- "이런 데 괜찮고, 배 아저씨가 누를 때 어때요?"
- "우리 친구 일단은 다리를 좀 필 거예요, 아저씨가."
```

특징:
- 구어체 표현 ("어때요", "괜찮아요")
- 반복적 확인 질문
- 짧고 간결한 답변 ("네", "아니요")
- 상황 설명과 지시사항

### 4. NER 태깅 대상

현재 데이터에서는 `pred_answer`, `gold_answer_code`, `gold_answer_nl` 모두 빈 JSON (`{}`)으로 되어 있으나, 실제 라벨링 시 다음과 같은 엔티티를 추출할 수 있습니다:

#### 주요 엔티티 타입
- **신체 부위**: 목, 배, 무릎, 다리, 손, 발
- **증상**: 출혈, 통증, 고름
- **행위**: 누르다, 돌리다, 펴다
- **의료 장비**: 들것
- **인물**: 아저씨, 친구, 형

## 학습 데이터 변환 가이드

### 1. CSV → JSONL 변환

```python
import csv
import json

def convert_csv_to_jsonl(csv_file, output_file):
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        with open(output_file, 'w', encoding='utf-8') as out:
            for row in reader:
                # 텍스트 전처리
                text = row['sentence'].strip()

                # NER 결과 파싱 (현재는 빈 값)
                entities = json.loads(row['gold_answer_nl'] or '{}')

                # 대화형 포맷으로 변환
                messages = [
                    {
                        "role": "user",
                        "content": f"다음 텍스트에서 명명된 개체를 추출하여 JSON 형식으로 반환하세요.\n\n텍스트: {text}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(entities, ensure_ascii=False)
                    }
                ]

                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
```

### 2. 엔티티 매핑 예시

```json
{
  "신체부위": ["목", "배", "무릎", "다리", "손", "발"],
  "증상": ["출혈", "고름"],
  "의료장비": ["들것"],
  "인물": ["아저씨", "친구", "형"]
}
```

## 데이터 활용 시 주의사항

1. **개인정보**: 실제 응급 상황 데이터이므로 개인정보 보호 필요
2. **시간 정보**: `start`, `end` 필드로 대화의 시간적 맥락 파악 가능
3. **화자 정보**: 화자별로 다른 언어 패턴과 역할 구분 필요
4. **도메인 특성**: 의료/응급 도메인 특화 용어와 표현 이해 필요

## 추가 필요 작업

1. **라벨링 완성**: 현재 빈 값인 `gold_answer` 필드 채우기
2. **엔티티 타입 정의**: 도메인에 맞는 NER 태그 세트 정의
3. **가이드라인 작성**: 일관된 라벨링을 위한 상세 가이드라인
4. **품질 검증**: 라벨링 일관성 및 정확도 검증

## 관련 파일

- **코드표**: `data/20250911 소방청 코드표.xlsx` - 응급 상황 관련 코드 정의
- **오디오**: `data/20250911 NER 라벨링 데이터/audio/` - 원본 음성 파일
- **매핑**: `data/20250911 NER 라벨링 데이터/mapping.json` - 파일 매핑 정보

## 통계 정보

- 총 CSV 파일 수: 다수 (all_completed_exports 디렉토리)
- 평균 발화 수: 약 15-20개 per file
- 주요 화자: A (응급구조사), B/C (기타)
- 평균 발화 길이: 5-50자