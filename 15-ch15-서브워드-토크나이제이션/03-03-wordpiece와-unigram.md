파일이 생성되었습니다. 세션 15.3의 주요 내용:

- **WordPiece**: 우도 기반 병합 점수(freq(xy)/freq(x)×freq(y))와 탐욕적 최장 일치 인코딩, `##` 접두사

> 📊 **그림 2**: WordPiece 인코딩 — 탐욕적 최장 일치

```mermaid
flowchart LR
    A["입력: unhappily"] --> B["어휘에서 최장 매치 탐색"]
    B --> C["un 매치"]
    C --> D["나머지: happily"]
    D --> E["##happi 매치"]
    E --> F["나머지: ly"]
    F --> G["##ly 매치"]
    G --> H["결과: un, ##happi, ##ly"]
```


> 📊 **그림 1**: WordPiece 병합 학습 과정

```mermaid
flowchart TD
    A["초기 어휘: 모든 문자"] --> B["후보 쌍 생성"]
    B --> C["우도 점수 계산<br/>freq(xy) / freq(x)*freq(y)"]
    C --> D{"최고 점수 쌍 선택"}
    D --> E["어휘에 병합 토큰 추가"]
    E --> F{"목표 어휘 크기 도달?"}
    F -->|아니오| B
    F -->|예| G["최종 어휘 완성"]
```

- **Unigram**: 하향식 삭제, 비터비 알고리즘, 서브워드 정규화(다중 토큰화)

> 📊 **그림 4**: Unigram 인코딩 — 비터비 알고리즘

```mermaid
flowchart LR
    A["입력 단어"] --> B["가능한 모든<br/>분할 후보 생성"]
    B --> C["각 분할의<br/>로그 확률 합산"]
    C --> D["비터비로<br/>최적 경로 탐색"]
    D --> E["최고 확률<br/>분할 선택"]
```


> 📊 **그림 3**: Unigram 모델 학습 과정 — 하향식 삭제

```mermaid
flowchart TD
    A["큰 초기 어휘 구성"] --> B["EM 알고리즘으로<br/>서브워드 확률 추정"]
    B --> C["각 서브워드의<br/>손실 기여도 계산"]
    C --> D["손실 영향 가장 작은<br/>서브워드 제거"]
    D --> E{"목표 어휘 크기 도달?"}
    E -->|아니오| B
    E -->|예| F["최종 어휘 완성"]
```

- **3대 알고리즘 종합 비교표**: 접근 방향, 학습 기준, 인코딩 방식, 대표 모델

> 📊 **그림 5**: BPE vs WordPiece vs Unigram 비교

```mermaid
graph TD
    subgraph BPE
        B1["상향식 병합"] --> B2["빈도 기반 점수"]
        B2 --> B3["결정적 인코딩"]
        B3 --> B4["GPT, LLaMA"]
    end
    subgraph WordPiece
        W1["상향식 병합"] --> W2["우도 기반 점수"]
        W2 --> W3["최장 일치 인코딩"]
        W3 --> W4["BERT"]
    end
    subgraph Unigram
        U1["하향식 삭제"] --> U2["손실 기반 점수"]
        U2 --> U3["확률적 인코딩"]
        U3 --> U4["T5, ALBERT"]
    end
```

- Mermaid 다이어그램 5개, `run:python` 블록 3개, 실습 코드 2개(SimpleWordPiece, SimpleUnigram)