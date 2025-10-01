# 재무제표 분석 시스템 v2

DART에서 제공하는 텍스트 형태의 재무제표를 분석하여, 사용자의 자연어 질문에 답변하는 AI 챗봇 시스템입니다.

## 🎯 프로젝트 개요

### 핵심 기능
1. **텍스트 파일 파싱 및 SQLite 데이터베이스 구축**
2. **자연어 질문을 SQL로 변환하여 DB 조회 (Text2SQL)**
   - 벡터 기반 고유명사 검색으로 회사명/항목명 매칭 정확도 향상
3. **재무 외 질문은 Tavily 웹 검색으로 답변**
4. **LangGraph를 사용하여 질문의 유형에 따라 3가지 경로로 동적 라우팅**
   - No Retrieval: 일반 상식 질문
   - Single-shot RAG: 단일 데이터 조회
   - Iterative RAG: 복잡한 비교 분석
5. **단일 대화 세션 내에서 대화 기록 관리**
6. **Gradio 기반의 간단한 채팅 UI**

### v2 개선사항
- ✨ **벡터 기반 고유명사 검색**: OpenAI Embeddings + InMemoryVectorStore를 사용하여 회사명과 재무항목명을 벡터화하여 저장. 유사도 기반 검색으로 오타나 다양한 표현에도 정확한 매칭 가능
- 🔧 **LangGraph StateGraph 기반 Text2SQL**: 쿼리 생성 → 실행 → 답변 생성의 파이프라인을 StateGraph로 구현
- 💬 **대화 컨텍스트 유지**: MemorySaver를 사용한 Short-term Memory로 이전 대화 내용을 기억
- 🎯 **Adaptive RAG**: 질문의 복잡도에 따라 적절한 처리 방법 자동 선택

## 🛠 기술 스택

- **언어**: Python
- **핵심 로직**: LangChain, LangGraph
- **LLM**: GPT-4o-mini
- **임베딩**: text-embedding-3-large
- **데이터베이스**: SQLite
- **메모리**: LangGraph Short-term Memory (MemorySaver)
- **웹 검색**: Tavily Search API
- **UI**: Gradio

## 📁 프로젝트 구조

```
financial_analysis_poc_v2/
├── data/
│   ├── balance_sheets/          # 재무상태표 데이터 (*.txt)
│   ├── income_statements/       # 손익계산서 데이터 (*.txt)
│   ├── cash_flow_statements/    # 현금흐름표 데이터 (*.txt)
│   └── equity_statements/       # 자본변동표 데이터 (*.txt)
├── main.py                      # Gradio UI 및 전체 애플리케이션 실행
├── database.py                  # SQLite DB 초기화 및 스키마 정의
├── parser.py                    # 텍스트 파일 파싱 및 DB 저장
├── graph.py                     # LangGraph 워크플로우(StateGraph) 정의
├── tools.py                     # Text2SQL, Tavily, 벡터스토어 등 도구 정의
├── requirements.txt             # 필요한 파이썬 패키지 목록
├── .env.template                # 환경 변수 템플릿
├── financial_data.db            # 생성될 SQLite DB 파일
└── README.md                    # 프로젝트 설명서
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.template` 파일을 `.env`로 복사하고 API 키를 입력합니다:

```bash
cp .env.template .env
```

`.env` 파일 내용:
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. 데이터 준비

`data/` 폴더에 재무제표 데이터 파일(*.txt)을 넣습니다:
- `data/balance_sheets/`: 재무상태표
- `data/income_statements/`: 손익계산서
- `data/cash_flow_statements/`: 현금흐름표
- `data/equity_statements/`: 자본변동표

### 4. 애플리케이션 실행

```bash
python main.py
```

웹 브라우저에서 `http://localhost:7860` 접속

## 💡 사용 예시

### 예시 질문

1. **일반 상식 질문** (No Retrieval)
   - "재무상태표가 뭐야?"
   - "자본변동표란?"

2. **단순 데이터 조회** (Single-shot RAG)
   - "SK텔레콤의 2025년 상반기 매출액은 얼마야?"
   - "삼성전자의 영업이익은?"

3. **복잡한 비교 분석** (Iterative RAG)
   - "SKT와 KT의 2025년 상반기 영업이익률을 비교하고 그 차이의 원인을 분석해줘."
   - "삼성전자와 SK하이닉스의 재무 건전성을 비교해줘."

4. **웹 검색**
   - "요즘 AI 기술 트렌드에 대해 알려줘."
   - "최근 반도체 시장 동향은?"

## 🔧 주요 구성 요소

### 1. Database (database.py)
- SQLite 데이터베이스 초기화
- 4개 테이블: balance_sheet, income_statement, cash_flow_statement, statement_of_changes_in_equity
- 회사명 및 재무항목명 추출 기능

### 2. Parser (parser.py)
- TSV 파일 파싱
- 다중 인코딩 지원 (UTF-8, CP949, EUC-KR, UTF-16)
- 데이터 정규화 및 DB 삽입

### 3. Tools (tools.py)
- **벡터스토어 기반 고유명사 검색**: 회사명과 재무항목명을 벡터화하여 유사도 검색
- **Text2SQL**: LangGraph StateGraph 기반 SQL 쿼리 생성 및 실행
- **Tavily 웹 검색**: 재무 외 정보 검색

### 4. Graph (graph.py)
- **Adaptive RAG 워크플로우**: 질문 분석 → 라우팅 → RAG → 답변 생성
- **3가지 처리 경로**:
  - No Retrieval: LLM 자체 지식으로 답변
  - Single-shot RAG: 한 번의 도구 호출로 답변
  - Iterative RAG: 여러 도구를 순차적으로 사용하여 복잡한 질문 해결
- **Short-term Memory**: MemorySaver를 사용한 대화 기록 관리

### 5. Main (main.py)
- Gradio UI 구성
- 데이터 초기화 및 시스템 실행

## 📊 데이터베이스 스키마

### balance_sheet (재무상태표)
- 회사명, 결산기준일, 항목명, 당기_반기말, 전기말, 전전기말 등

### income_statement (손익계산서)
- 회사명, 결산기준일, 항목명, 당기_반기_3개월, 당기_반기_누적, 전기, 전전기 등

### cash_flow_statement (현금흐름표)
- 회사명, 결산기준일, 항목명, 당기_반기말, 전기_반기말, 전기, 전전기 등

### statement_of_changes_in_equity (자본변동표)
- 회사명, 결산기준일, 항목명, 당기, 전기, 전전기 등

## 🔍 Text2SQL 처리 흐름

1. **사용자 질문 입력**
2. **고유명사 벡터 검색**: 질문에서 회사명/항목명 추출 및 벡터스토어에서 유사 검색
3. **SQL 쿼리 생성**: LLM이 고유명사 정보를 참고하여 SQL 쿼리 생성
4. **쿼리 실행**: SQLite에서 쿼리 실행
5. **답변 생성**: LLM이 쿼리 결과를 바탕으로 자연어 답변 생성

## 🤝 기여

이 프로젝트는 LangChain과 LangGraph를 활용한 재무 데이터 분석 시스템의 PoC(Proof of Concept)입니다.

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 🙏 참고 자료

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Tavily AI Search](https://tavily.com/)

