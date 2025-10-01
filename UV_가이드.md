# UV 가상환경 설정 가이드

## 📦 UV란?

UV는 Rust로 작성된 초고속 Python 패키지 관리자입니다. pip보다 10-100배 빠른 패키지 설치 속도를 자랑합니다.

## 🚀 설치 및 실행

### 1. UV 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew
brew install uv

# 설치 확인
uv --version
```

### 2. 가상환경 생성 및 패키지 설치

```bash
# 프로젝트 디렉토리로 이동
cd /Users/1107625/dev/repositories/scripts/2509_LLMMVP/MVP/financial_analysis_poc_v2

# 가상환경 생성 및 패키지 설치 (한 번에!)
uv sync

# 또는 수동으로
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 3. 애플리케이션 실행

```bash
# 가상환경 활성화 상태에서
python main.py

# 또는 uv를 통해 직접 실행
uv run python main.py
```

## 📝 주요 명령어

### 패키지 관리

```bash
# 패키지 추가
uv add langchain

# 개발 패키지 추가
uv add --dev pytest

# 패키지 제거
uv remove langchain

# 패키지 목록 확인
uv pip list

# 의존성 동기화 (pyproject.toml 기준)
uv sync
```

### 가상환경 관리

```bash
# 가상환경 생성
uv venv

# Python 버전 지정하여 가상환경 생성
uv venv --python 3.11

# 가상환경 활성화
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# 가상환경 비활성화
deactivate
```

### 프로젝트 실행

```bash
# uv run을 사용하면 자동으로 가상환경에서 실행
uv run python main.py

# 스크립트 직접 실행
uv run --script main.py

# 모듈로 실행
uv run -m parser
```

## 🔧 pyproject.toml 구조

```toml
[project]
name = "financial-analysis-poc-v2"
version = "2.0.0"
description = "재무제표 분석 시스템 v2"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3.27",
    "langgraph>=0.6.8",
    "langchain-openai>=0.3.33",
    "langchain-community>=0.3.30",
    "beautifulsoup4>=4.14.2",
    "tavily-python>=0.7.12",
    "gradio>=5.47.2",
    "python-dotenv>=1.1.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.7.0",
    "ruff>=0.0.287",
]
```

## ⚡️ UV의 장점

### 1. 속도
```bash
# pip
time pip install langchain
# 약 30-60초

# uv
time uv pip install langchain
# 약 3-5초 (10-20배 빠름!)
```

### 2. 의존성 해결
- 더 정확하고 빠른 의존성 해결
- uv.lock 파일로 재현 가능한 빌드

### 3. 통합 도구
```bash
# 기존 방식
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# uv 방식
uv sync  # 끝!
```

## 📊 성능 비교

| 작업 | pip | uv | 속도 향상 |
|------|-----|----|----|
| 패키지 설치 | 30s | 3s | 10x |
| 가상환경 생성 | 5s | 0.5s | 10x |
| 의존성 해결 | 15s | 1s | 15x |

## 🔄 기존 프로젝트 마이그레이션

### requirements.txt에서 변환

```bash
# requirements.txt가 있는 경우
uv pip compile requirements.txt -o requirements.lock

# 또는 pyproject.toml 생성
uv init --name financial-analysis-poc-v2
# pyproject.toml 수동 편집
uv sync
```

### pip에서 uv로 전환

```bash
# 1. 기존 가상환경 제거
rm -rf venv/

# 2. uv로 새 가상환경 생성
uv venv

# 3. 패키지 설치
uv sync
```

## 🛠 고급 사용법

### 특정 Python 버전 사용

```bash
# Python 3.11 사용
uv venv --python 3.11

# Python 3.12 사용
uv venv --python 3.12
```

### 오프라인 설치

```bash
# 캐시 확인
uv cache dir

# 캐시 정리
uv cache clean

# 오프라인 모드
uv sync --offline
```

### Lock 파일 관리

```bash
# uv.lock 생성/업데이트
uv lock

# lock 파일 기준으로 설치
uv sync --frozen

# lock 파일 없이 설치
uv sync --no-lock
```

## 🐛 문제 해결

### UV가 설치되지 않는 경우

```bash
# PATH 확인
echo $PATH

# UV 경로 추가 (macOS/Linux)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 패키지 충돌

```bash
# 캐시 정리
uv cache clean

# 가상환경 재생성
rm -rf .venv
uv venv
uv sync
```

### Python 버전 문제

```bash
# 사용 가능한 Python 버전 확인
uv python list

# 특정 버전 설치
uv python install 3.11
```

## 📚 참고 자료

- [UV 공식 문서](https://docs.astral.sh/uv/)
- [UV GitHub](https://github.com/astral-sh/uv)
- [빠른 시작 가이드](https://docs.astral.sh/uv/getting-started/)

## 🎯 권장 워크플로우

```bash
# 1. 프로젝트 시작
cd financial_analysis_poc_v2

# 2. 환경 설정 (최초 1회)
uv sync

# 3. 환경 변수 설정
cp .env.template .env
# .env 파일 편집

# 4. 애플리케이션 실행
uv run python main.py

# 5. 개발 중 패키지 추가
uv add <package-name>

# 6. 테스트 (개발 의존성)
uv add --dev pytest
uv run pytest
```

## ✅ 체크리스트

- [ ] UV 설치 완료
- [ ] `uv sync` 실행 완료
- [ ] `.env` 파일 설정 완료
- [ ] `uv run python main.py` 실행 성공
- [ ] 브라우저에서 http://localhost:7860 접속 확인

축하합니다! UV를 사용한 개발 환경이 준비되었습니다! 🎉

