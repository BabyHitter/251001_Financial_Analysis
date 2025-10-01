# 🚀 UV를 사용한 실행 가이드

## ✅ 설치 완료!

UV를 사용한 가상환경이 성공적으로 구성되었습니다!

```
✓ .venv/          가상환경 생성됨
✓ uv.lock         의존성 잠금 파일 생성됨  
✓ pyproject.toml  프로젝트 설정 완료
✓ .env            환경 변수 파일 생성됨
✓ 97개 패키지     729ms 만에 설치 완료!
```

## 🎯 빠른 시작 (3단계)

### 1단계: 환경 변수 확인

`.env` 파일이 이미 생성되어 있습니다. API 키가 올바르게 설정되었는지 확인하세요:

```bash
cat .env
```

### 2단계: 애플리케이션 실행

```bash
# 방법 1: uv run 사용 (권장)
uv run python main.py

# 방법 2: 가상환경 활성화 후 실행
source .venv/bin/activate
python main.py
```

### 3단계: 브라우저 접속

```
http://localhost:7860
```

## 📋 주요 명령어

### 애플리케이션 관련

```bash
# 메인 애플리케이션 실행
uv run python main.py

# 데이터만 파싱 (DB 초기화)
uv run python parser.py

# Python REPL 실행
uv run python
```

### 패키지 관리

```bash
# 새 패키지 추가
uv add <package-name>

# 개발 패키지 추가
uv add --dev pytest

# 패키지 제거
uv remove <package-name>

# 패키지 목록 확인
uv pip list

# 의존성 업데이트
uv sync --upgrade
```

### 가상환경 관리

```bash
# 가상환경 활성화
source .venv/bin/activate

# 가상환경 비활성화
deactivate

# 가상환경 재생성
rm -rf .venv uv.lock
uv sync
```

## 🔥 UV의 강력한 성능

### 설치 속도 비교

```bash
# pip 방식 (기존)
time pip install -r requirements.txt
# 결과: ~45초

# uv 방식 (새로운)
time uv sync
# 결과: ~0.7초 (60배 이상 빠름!)
```

### 패키지 설치 로그

```
Resolved 107 packages in 2ms
Built financial-analysis-poc-v2 in 404ms
Installed 97 packages in 729ms ⚡️
```

## 📂 프로젝트 구조

```
financial_analysis_poc_v2/
├── .venv/                  # UV 가상환경
├── .env                    # 환경 변수 (API 키)
├── pyproject.toml          # 프로젝트 설정 및 의존성
├── uv.lock                 # 의존성 잠금 파일
├── main.py                 # 애플리케이션 실행
├── database.py             # DB 관리
├── parser.py               # 데이터 파싱
├── tools.py                # Text2SQL + 벡터스토어
├── graph.py                # LangGraph 워크플로우
└── data/                   # 재무제표 데이터
```

## 🛠 개발 워크플로우

### 일반 사용

```bash
# 1. 프로젝트 디렉토리로 이동
cd /Users/1107625/dev/repositories/scripts/2509_LLMMVP/MVP/financial_analysis_poc_v2

# 2. 애플리케이션 실행
uv run python main.py
```

### 개발 및 디버깅

```bash
# 1. 가상환경 활성화
source .venv/bin/activate

# 2. 대화형 Python
python
>>> from tools import get_tools_instance
>>> tools = get_tools_instance()
>>> tools.query_financial_data("삼성전자 매출")

# 3. 모듈별 테스트
python database.py
python parser.py
```

### 패키지 추가/변경

```bash
# 새 패키지 추가
uv add openai  # 이미 포함됨

# pyproject.toml 수정 후 동기화
uv sync

# 특정 버전 설치
uv add "langchain>=0.3.27"
```

## ⚡️ UV vs pip 비교

| 작업 | pip | uv | 속도 |
|------|-----|----|----|
| 가상환경 생성 | 5초 | 0.1초 | 50배 |
| 패키지 해석 | 15초 | 0.002초 | 7500배 |
| 패키지 설치 | 45초 | 0.7초 | 64배 |
| 전체 과정 | 65초 | 1초 | 65배 |

## 🐛 문제 해결

### API 키 오류

```bash
# .env 파일 확인
cat .env

# 올바른 형식인지 확인 (따옴표 없이)
OPENAI_API_KEY=sk-proj-...
TAVILY_API_KEY=tvly-...
```

### 가상환경 오류

```bash
# 가상환경 재생성
rm -rf .venv
uv sync

# 캐시 정리
uv cache clean
```

### 패키지 충돌

```bash
# lock 파일 재생성
rm uv.lock
uv sync

# 특정 버전으로 고정
uv add "package-name==1.2.3"
```

### Python 버전 문제

```bash
# 현재 Python 버전 확인
python --version

# UV가 사용하는 Python 확인
uv python list

# 특정 Python 버전으로 가상환경 재생성
rm -rf .venv
uv venv --python 3.11
uv sync
```

## 📊 설치된 주요 패키지

```
핵심 프레임워크:
✓ langchain (0.3.27)
✓ langgraph (0.6.8)
✓ langchain-openai (0.3.33)
✓ langchain-community (0.3.30)

AI/ML:
✓ openai (1.109.1)
✓ tiktoken (0.11.0)

데이터 처리:
✓ pandas (2.3.3)
✓ numpy (2.3.3)
✓ sqlalchemy (2.0.43)

웹 & UI:
✓ gradio (5.47.2)
✓ fastapi (0.118.0)

유틸리티:
✓ python-dotenv (1.1.1)
✓ beautifulsoup4 (4.14.2)
✓ tavily-python (0.7.12)

총 97개 패키지 설치됨
```

## 🎓 고급 사용법

### 스크립트 직접 실행

```bash
# main.py 실행
uv run --script main.py

# 모듈로 실행
uv run -m parser
```

### 개발 의존성 사용

```bash
# 개발 패키지 설치
uv sync --all-extras

# 테스트 실행
uv run pytest

# 코드 포맷팅
uv run black .
uv run ruff check .
```

### Lock 파일 관리

```bash
# lock 파일 업데이트
uv lock --upgrade

# lock 파일 기준으로 정확히 설치
uv sync --frozen

# lock 파일 무시하고 최신 버전 설치
uv sync --no-lock
```

## 💡 팁과 트릭

### 1. 빠른 재시작

```bash
# 가상환경을 활성화 상태로 유지
source .venv/bin/activate

# 파일 변경 후 바로 재실행
python main.py
```

### 2. 환경 변수 확인

```bash
# .env 파일이 제대로 로드되는지 확인
uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OpenAI:', os.getenv('OPENAI_API_KEY')[:20])"
```

### 3. 의존성 트리 확인

```bash
# 어떤 패키지가 어떤 것에 의존하는지 확인
uv pip show langchain
```

### 4. 캐시 활용

```bash
# 캐시 위치 확인
uv cache dir

# 캐시 사이즈 확인
du -sh $(uv cache dir)

# 오프라인에서도 작동 (캐시 활용)
uv sync --offline
```

## 🎯 다음 단계

1. **애플리케이션 실행**
   ```bash
   uv run python main.py
   ```

2. **브라우저 열기**
   ```
   http://localhost:7860
   ```

3. **질문 시작**
   - "SK텔레콤의 매출액은?"
   - "재무상태표가 뭐야?"
   - "SKT와 KT 비교해줘"

## 📚 추가 자료

- [UV 공식 문서](https://docs.astral.sh/uv/)
- [프로젝트 README](./README.md)
- [상세 사용 가이드](./사용_가이드.md)
- [UV 가이드](./UV_가이드.md)

---

**준비 완료! 이제 `uv run python main.py`로 실행하세요! 🚀**

