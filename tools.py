import os
import ast
import re
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from tavily import TavilyClient
from database import db as financial_db

# 환경 변수 로드
load_dotenv()


def query_as_list(db_instance, query):
    """DB 쿼리 결과를 리스트로 변환합니다."""
    res = db_instance.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


class State(TypedDict):
    """Text2SQL 상태를 정의합니다."""
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """생성된 SQL 쿼리."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


class FinancialAnalysisTools:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.openai_api_key
        )
        
        # SQL 데이터베이스 연결
        self.db = SQLDatabase.from_uri("sqlite:///financial_data.db")
        
        # Tavily 클라이언트 초기화
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        
        # 벡터스토어 초기화 (고유명사 처리용)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.entity_retriever = None
        
        # 고유명사 벡터스토어 구축
        self._build_entity_vector_store()
        
        # Text2SQL 그래프 초기화
        self.text2sql_graph = self._build_text2sql_graph()
    
    def _build_entity_vector_store(self):
        """회사명과 재무항목명을 벡터스토어에 저장합니다."""
        try:
            # DB에서 회사명과 항목명 추출
            companies = financial_db.get_all_companies()
            items = financial_db.get_all_items()
            
            print(f"회사명 {len(companies)}개, 재무항목 {len(items)}개를 벡터스토어에 저장 중...")
            
            # 벡터스토어에 추가
            all_entities = companies + items
            if all_entities:
                self.vector_store.add_texts(all_entities)
                self.entity_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
                print("고유명사 벡터스토어 구축 완료")
            else:
                print("경고: 벡터스토어에 추가할 데이터가 없습니다.")
        except Exception as e:
            print(f"벡터스토어 구축 중 오류: {e}")
            self.entity_retriever = None
    
    def search_entities(self, query: str) -> str:
        """질문에서 고유명사를 검색합니다."""
        if not self.entity_retriever:
            return ""
        
        try:
            docs = self.entity_retriever.invoke(query)
            entities = [doc.page_content for doc in docs]
            return "\n".join(entities)
        except Exception as e:
            print(f"고유명사 검색 중 오류: {e}")
            return ""
    
    def _build_text2sql_graph(self) -> StateGraph:
        """Text2SQL 그래프를 구축합니다 (고유명사 처리 포함)."""
        
        # SQL 쿼리 생성 프롬프트 (고유명사 정보 포함)
        query_prompt_template = ChatPromptTemplate.from_template("""
Given an input question, create a syntactically correct {dialect} query to run to help find the answer. 

**CRITICAL: LIMIT Rules - READ THIS CAREFULLY!**
The default {top_k} is ONLY for simple queries. Many queries need LIMIT 100!

Check if the question contains ANY of these keywords:
- Korean: "모두", "전부", "모든", "추출", "전체", "모두 조회", "모두 추출"
- English: "all", "all companies", "extract all", "show all"
- Filtering context: "~이면서 ~인 기업" (usually means "all matching companies")

**Decision Logic:**
1. If ANY keyword above appears → LIMIT 100 (NOT {top_k}!)
2. If user specifies exact number ("상위 5개", "top 10") → Use that number
3. ONLY if neither above → Use LIMIT {top_k}

**Examples:**
- "모두 추출해줘" → LIMIT 100 ✅
- "영업이익률 20% 이상인 기업 모두" → LIMIT 100 ✅
- "~이면서 ~인 회사 추출" → LIMIT 100 ✅
- "상위 5개 기업" → LIMIT 5 ✅
- "삼성전자의 매출액은?" → LIMIT {top_k} ✅

**CRITICAL: Range Conditions (범위 조건)**
When user specifies ranges like "100억 이상 1000억 미만", "X 이상 Y 미만", "X ~ Y":
- ALWAYS use BOTH lower bound (>=) AND upper bound (<)
- Example: "매출액 100억 이상 1000억 미만"
  → `CAST(REPLACE(매출액, ',', '') AS REAL) >= 10000000000`
  → `AND CAST(REPLACE(매출액, ',', '') AS REAL) < 100000000000`
- "이상" = >= (inclusive), "미만" = < (exclusive)
- "초과" = > (exclusive), "이하" = <= (inclusive)
- NEVER forget the upper bound! This is critical for accurate filtering!

You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. 
Be careful to not query for columns that do not exist.

Also, pay attention to which column is in which table.
Only use the following tables:
{table_info}

Entity names and their relationships to consider:
{entity_info}

## Matching Guidelines
- Use exact matches when comparing entity names
- Check for historical name variations if available
- Apply case-sensitive matching for official names
- Handle both Korean and English entity names when present

## CRITICAL: Company Name Mapping (대표 회사명 우선)
**When user asks about major companies using abbreviations or English names:**

**Telecom Companies (통신사):**
- "kt", "KT", "케이티" → Use EXACT match: 회사명 = '케이티' (KT Corp main company)
  - NOT "KT밀리의서재", "KT나스미디어" (these are subsidiaries)
  - Use WHERE 회사명 = '케이티' (= operator, not LIKE)
- "skt", "SKT", "sk텔레콤", "에스케이텔레콤" → 회사명 = 'SK텔레콤'
- "lgu+", "LG유플러스", "엘지유플러스" → 회사명 = 'LG유플러스'

**Electronics/IT:**
- "삼성", "삼성전자" → 회사명 = '삼성전자' (not 삼성SDI, 삼성E&A, etc.)
- "lg전자", "LG" → 회사명 = '엘지전자' or 'LG전자'
- "sk하이닉스", "하이닉스" → 회사명 = 'SK하이닉스'

**Important Rules:**
1. When user mentions just "KT" or "kt", they mean the main company "케이티", NOT subsidiaries
2. Always use = (equals) operator for company names, NOT LIKE
3. Only use LIKE for 항목명 (item names), not for 회사명 (company names)
4. If multiple companies match, prioritize the main/parent company

**Example Queries:**
- "kt의 영업이익" → WHERE 회사명 = '케이티' AND 항목명 LIKE '%영업이익%'
- "삼성전자 매출" → WHERE 회사명 = '삼성전자' AND 항목명 LIKE '%매출%'
- "SK텔레콤 순이익" → WHERE 회사명 = 'SK텔레콤' AND 항목명 LIKE '%순이익%'

## CRITICAL: Financial Term Mapping (재무용어 매핑)
**When user asks about financial terms, map them to actual column names in the database:**

**Balance Sheet Terms (재무상태표):**
- "자산" → Look for '자산총계' or '자산 총계'
  - Example: "삼성전자 자산은?" → WHERE 항목명 LIKE '%자산총계%' OR 항목명 = '자산총계'
- "부채" → Look for '부채총계' or '부채 총계'
- "자본" → Look for '자본총계' or '자본 총계'
- "유동자산" → '유동자산' or starts with '유동자산'
- "비유동자산" → '비유동자산' or starts with '비유동자산'

**Income Statement Terms (손익계산서):**
- "매출", "매출액" → **CRITICAL: Industry-specific mapping!**
  - **General Manufacturing (제조업: 삼성전자, SK하이닉스, etc.)**: Use '매출액'
    - WHERE 항목명 LIKE '%매출액%' OR 항목명 LIKE 'I. 매출%'
  - **Finance/Telecom (금융/통신: SK텔레콤, 케이티, LG유플러스, etc.)**: Use '영업수익'
    - WHERE 항목명 LIKE '%영업수익%' OR 항목명 = '영업수익'
  - **IMPORTANT**: Try BOTH patterns if company type is unclear:
    - WHERE (항목명 LIKE '%매출액%' OR 항목명 LIKE '%영업수익%')
  
- "영업이익" → Look for '영업이익' (including variations like '영업이익(손실)')
  - WHERE 항목명 LIKE '%영업이익%'

- "순이익", "당기순이익" → **CRITICAL: Company-specific variations!**
  - **케이티, LG유플러스**: "반기순이익"
  - **SK텔레콤**: "당기순이익" (반기순이익 없음!)
  - **삼성전자, SK하이닉스**: "반기순이익"
  - **ALWAYS use ALL patterns to catch all companies**:
    - WHERE (항목명 LIKE '%반기순이익%' OR 항목명 LIKE '%당기순이익%' OR 항목명 = '당기순이익' OR 항목명 LIKE '%순이익%')
  - **NEVER query just '반기순이익' - SK텔레콤 will fail!**
  - **NEVER query just '당기순이익' - 케이티/LG유플러스 will fail!**

- "매출총이익" → Look for '매출총이익'
  - WHERE 항목명 LIKE '%매출총이익%'

**Important Pattern Matching Rules:**
1. For "총계" items (자산, 부채, 자본), use: `항목명 LIKE '%[term]총계%'`
2. For income items, some have Roman numerals: `항목명 LIKE 'I. 매출%' OR 항목명 LIKE 'Ⅰ. 매출%' OR 항목명 LIKE '%매출액%'`
3. Some items have variations (손실): `항목명 LIKE '%영업이익%'` covers both '영업이익' and '영업이익(손실)'

**Complete Examples:**
```sql
-- "삼성전자 자산은?"
SELECT 회사명, 항목명, 당기_반기말 
FROM balance_sheet 
WHERE 회사명 = '삼성전자' 
  AND (항목명 LIKE '%자산총계%' OR 항목명 = '자산총계')

-- "케이티 부채는?"
SELECT 회사명, 항목명, 당기_반기말
FROM balance_sheet
WHERE 회사명 = '케이티'
  AND (항목명 LIKE '%부채총계%' OR 항목명 = '부채총계')

-- "SK텔레콤 매출은?" (통신사 → 영업수익)
SELECT 회사명, 항목명, 당기_반기_누적
FROM income_statement
WHERE 회사명 = 'SK텔레콤'
  AND (항목명 LIKE '%영업수익%' OR 항목명 LIKE '%매출액%')

-- "케이티 매출액, 영업이익, 순이익은?" (다중 항목 조회)
SELECT 회사명, 항목명, 당기_반기_누적
FROM income_statement
WHERE 회사명 = '케이티'
  AND (항목명 LIKE '%영업수익%' OR 항목명 LIKE '%매출액%' 
       OR 항목명 LIKE '%영업이익%' 
       OR 항목명 LIKE '%반기순이익%' OR 항목명 LIKE '%당기순이익%' OR 항목명 LIKE '%순이익%')

-- "SK텔레콤 순이익은?" (당기순이익 사용! - 반기순이익 없음)
SELECT 회사명, 항목명, 당기_반기_누적
FROM income_statement
WHERE 회사명 = 'SK텔레콤'
  AND (항목명 LIKE '%반기순이익%' OR 항목명 LIKE '%당기순이익%' OR 항목명 = '당기순이익' OR 항목명 LIKE '%순이익%')

-- "삼성전자 순이익은?" (반기순이익 사용)
SELECT 회사명, 항목명, 당기_반기_누적
FROM income_statement
WHERE 회사명 = '삼성전자'
  AND (항목명 LIKE '%반기순이익%' OR 항목명 LIKE '%당기순이익%' OR 항목명 LIKE '%순이익%')
```

## 🚨 CRITICAL: Ambiguous Company Name Handling 🚨
**Problem:** User asks "sk의 매출액" → 25 companies match (SK, SKC, SK텔레콤, SK하이닉스, etc.)

**Solution: ALWAYS use EXACT company name matching!**
- **NEVER use `LIKE '%sk%'`** - this matches 25 companies and creates confusing results!
- **ALWAYS use exact match**: `WHERE 회사명 = 'SK텔레콤'` ✅
- **Check entity_info for exact company name** - it provides the correct full name

**Common Ambiguous Cases:**
1. **"sk" alone** → 25 possible companies (SK, SKC, SK텔레콤, SK하이닉스, SK네트웍스, SK스퀘어, etc.)
2. **"삼성" alone** → Multiple (삼성전자, 삼성중공업, 삼성SDI, etc.)
3. **"lg" alone** → Multiple (LG전자, LG유플러스, LG화학, etc.)

**What to do:**
- If entity_info provides exact company name → USE IT!
- If user query is ambiguous and no entity_info → Default to most common:
  - "sk" → "SK텔레콤" or "SK하이닉스" (depending on context)
  - "삼성" → "삼성전자"
  - "lg" → "LG전자"

**Examples:**
```sql
-- ❌ BAD (matches 25 companies!)
SELECT * FROM income_statement WHERE 회사명 LIKE '%sk%'

-- ✅ GOOD (exact match)
SELECT * FROM income_statement WHERE 회사명 = 'SK텔레콤'

-- ✅ ACCEPTABLE (multiple specific companies)
SELECT * FROM income_statement 
WHERE 회사명 IN ('SK텔레콤', 'SK하이닉스', 'SK이노베이션')
```

## CRITICAL: Financial Ratio Calculation (재무비율 계산)
**The database does NOT have ratio columns. You must calculate them using SQL.**

**Common Financial Ratios and How to Calculate:**

**1. 영업이익률 (Operating Profit Margin):**
- Formula: (영업이익 / 매출액) × 100
- SQL Approach: Query BOTH 영업이익 and 매출액, then calculate in SQL or let LLM calculate

**2. 순이익률 (Net Profit Margin):**
- Formula: (순이익 / 매출액) × 100
- **CRITICAL**: For current 반기 data, use '반기순이익' not '당기순이익'
- **CRITICAL**: For telecom companies, use '영업수익' not '매출액'
- SQL Approach: Query BOTH 순이익 (반기순이익) and 매출액 (or 영업수익)

**3. ROE (Return on Equity) / 자기자본이익률:**
- Formula: (당기순이익 / 자본총계) × 100
- SQL Approach: Query 당기순이익 from income_statement AND 자본총계 from balance_sheet

**4. ROA (Return on Assets) / 총자산이익률:**
- Formula: (당기순이익 / 자산총계) × 100
- SQL Approach: Query 당기순이익 from income_statement AND 자산총계 from balance_sheet

**5. 부채비율 (Debt Ratio):**
- Formula: (부채총계 / 자본총계) × 100
- SQL Approach: Query BOTH from balance_sheet

**How to Handle Ratio Queries:**

**Option A: Use SQL JOIN and Calculate (RECOMMENDED)**
```sql
-- 영업이익률 계산 예시 (제조업 - 삼성전자)
SELECT 
    i_op.회사명,
    i_op.항목명 as 영업이익_항목,
    i_op.당기_반기_누적 as 영업이익,
    i_rev.항목명 as 매출_항목,
    i_rev.당기_반기_누적 as 매출액,
    ROUND(CAST(REPLACE(i_op.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL), 2) as 영업이익률
FROM income_statement i_op
JOIN income_statement i_rev ON i_op.회사명 = i_rev.회사명 
    AND i_op.결산기준일 = i_rev.결산기준일
WHERE i_op.회사명 = '삼성전자'
  AND i_op.항목명 LIKE '%영업이익%'
  AND (i_rev.항목명 LIKE '%매출액%' OR i_rev.항목명 LIKE 'I. 매출%')
LIMIT 1;

-- 영업이익률 계산 예시 (통신사 - 케이티) - 영업수익 사용!
SELECT 
    i_op.회사명,
    i_op.항목명 as 영업이익_항목,
    i_op.당기_반기_누적 as 영업이익,
    i_rev.항목명 as 매출_항목,
    i_rev.당기_반기_누적 as 영업수익,
    ROUND(CAST(REPLACE(i_op.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL), 2) as 영업이익률
FROM income_statement i_op
JOIN income_statement i_rev ON i_op.회사명 = i_rev.회사명 
    AND i_op.결산기준일 = i_rev.결산기준일
WHERE i_op.회사명 = '케이티'
  AND i_op.항목명 LIKE '%영업이익%'
  AND (i_rev.항목명 LIKE '%영업수익%' OR i_rev.항목명 LIKE '%매출액%')
LIMIT 1;

-- 순이익률 계산 예시 - 삼성전자 (반기순이익)
SELECT 
    i_net.회사명,
    i_net.항목명 as 순이익_항목,
    i_net.당기_반기_누적 as 순이익,
    i_rev.항목명 as 매출_항목,
    i_rev.당기_반기_누적 as 매출,
    ROUND(CAST(REPLACE(i_net.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL), 2) as 순이익률
FROM income_statement i_net
JOIN income_statement i_rev ON i_net.회사명 = i_rev.회사명 
    AND i_net.결산기준일 = i_rev.결산기준일
WHERE i_net.회사명 = '삼성전자'
  AND (i_net.항목명 LIKE '%반기순이익%' OR i_net.항목명 LIKE '%당기순이익%' OR i_net.항목명 LIKE '%순이익%')
  AND (i_rev.항목명 LIKE '%매출액%' OR i_rev.항목명 LIKE '%영업수익%')
LIMIT 1;

-- 순이익률 계산 예시 - SK텔레콤 (당기순이익 + 영업수익)
SELECT 
    i_net.회사명,
    i_net.항목명 as 순이익_항목,
    i_net.당기_반기_누적 as 순이익,
    i_rev.항목명 as 매출_항목,
    i_rev.당기_반기_누적 as 영업수익,
    ROUND(CAST(REPLACE(i_net.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL), 2) as 순이익률
FROM income_statement i_net
JOIN income_statement i_rev ON i_net.회사명 = i_rev.회사명 
    AND i_net.결산기준일 = i_rev.결산기준일
WHERE i_net.회사명 = 'SK텔레콤'
  AND (i_net.항목명 LIKE '%반기순이익%' OR i_net.항목명 LIKE '%당기순이익%' OR i_net.항목명 = '당기순이익')
  AND (i_rev.항목명 LIKE '%영업수익%' OR i_rev.항목명 LIKE '%매출액%')
LIMIT 1;
```

**Option B: Query Separately and Calculate in Answer**
```sql
-- For manufacturing companies (제조업)
-- Step 1: Get 영업이익
SELECT 회사명, 항목명, 당기_반기_누적 as 영업이익
FROM income_statement
WHERE 회사명 = '삼성전자' AND 항목명 LIKE '%영업이익%';

-- Step 2: Get 매출액  
SELECT 회사명, 항목명, 당기_반기_누적 as 매출액
FROM income_statement
WHERE 회사명 = '삼성전자' AND (항목명 LIKE '%매출액%' OR 항목명 LIKE 'I. 매출%');

-- For telecom companies (통신사)
-- Step 1: Get 영업이익
SELECT 회사명, 항목명, 당기_반기_누적 as 영업이익
FROM income_statement
WHERE 회사명 = '케이티' AND 항목명 LIKE '%영업이익%';

-- Step 2: Get 영업수익 (매출액 대신)
SELECT 회사명, 항목명, 당기_반기_누적 as 영업수익
FROM income_statement
WHERE 회사명 = '케이티' AND (항목명 LIKE '%영업수익%' OR 항목명 LIKE '%매출액%');

-- Step 3: Get 순이익 (반기순이익 or 당기순이익 - 회사마다 다름!)
SELECT 회사명, 항목명, 당기_반기_누적 as 순이익
FROM income_statement
WHERE 회사명 = '케이티' AND (항목명 LIKE '%반기순이익%' OR 항목명 LIKE '%당기순이익%' OR 항목명 LIKE '%순이익%');

-- SK텔레콤의 경우 (당기순이익만 있음)
SELECT 회사명, 항목명, 당기_반기_누적 as 순이익
FROM income_statement
WHERE 회사명 = 'SK텔레콤' AND (항목명 LIKE '%반기순이익%' OR 항목명 LIKE '%당기순이익%' OR 항목명 = '당기순이익');

-- Then in the answer generation, calculate: (영업이익 / 영업수익 * 100)
```

**Important Notes:**
1. When user asks for "영업이익률", "순이익률", "ROE", "ROA", or "부채비율", you MUST calculate it
2. Use SQL JOIN when possible (Option A) for efficiency
3. If JOIN is complex, query separately and calculate in answer (Option B)
4. Always show both the raw numbers AND the calculated ratio percentage
5. Format: "영업이익 11조원 / 매출액 50조원 = 영업이익률 22%"

**Ratio Query Examples:**
```sql
-- "삼성전자 영업이익률은?"
-- Use JOIN to get both values and calculate

-- "케이티 부채비율은?"  
SELECT 
    b_debt.회사명,
    b_debt.당기_반기말 as 부채,
    b_equity.당기_반기말 as 자본,
    ROUND(b_debt.당기_반기말 * 100.0 / b_equity.당기_반기말, 2) as 부채비율
FROM balance_sheet b_debt
JOIN balance_sheet b_equity ON b_debt.회사명 = b_equity.회사명
WHERE b_debt.회사명 = '케이티'
  AND b_debt.항목명 LIKE '%부채총계%'
  AND b_equity.항목명 LIKE '%자본총계%';
```

## CRITICAL: Multiple Conditions Across Tables (복합 조건 쿼리)
**When user asks for companies meeting multiple conditions from different tables:**

**Example: "영업이익 1000억 넘고 자산 1조 이상인 기업 추출해줘"**

This requires querying BOTH income_statement AND balance_sheet with JOIN.

**Approach: Use JOIN with WHERE conditions**
```sql
SELECT DISTINCT
    i.회사명,
    i.당기_반기_누적 as 영업이익,
    b.당기_반기말 as 자산총계
FROM income_statement i
JOIN balance_sheet b 
    ON i.회사명 = b.회사명 
    AND i.결산기준일 = b.결산기준일
WHERE i.항목명 LIKE '%영업이익%'
  AND i.당기_반기_누적 > 100000000000  -- 1000억
  AND b.항목명 LIKE '%자산총계%'
  AND b.당기_반기말 > 1000000000000  -- 1조
ORDER BY i.당기_반기_누적 DESC
LIMIT 20;
```

**More Complex Examples:**

```sql
-- "영업이익률 10% 이상이고 부채비율 50% 미만인 기업"
SELECT DISTINCT
    i_op.회사명,
    i_op.당기_반기_누적 as 영업이익,
    i_rev.당기_반기_누적 as 매출액,
    ROUND(i_op.당기_반기_누적 * 100.0 / i_rev.당기_반기_누적, 2) as 영업이익률,
    b_debt.당기_반기말 as 부채,
    b_equity.당기_반기말 as 자본,
    ROUND(b_debt.당기_반기말 * 100.0 / b_equity.당기_반기말, 2) as 부채비율
FROM income_statement i_op
JOIN income_statement i_rev 
    ON i_op.회사명 = i_rev.회사명 
    AND i_op.결산기준일 = i_rev.결산기준일
JOIN balance_sheet b_debt 
    ON i_op.회사명 = b_debt.회사명 
    AND i_op.결산기준일 = b_debt.결산기준일
JOIN balance_sheet b_equity 
    ON i_op.회사명 = b_equity.회사명 
    AND i_op.결산기준일 = b_equity.결산기준일
WHERE i_op.항목명 LIKE '%영업이익%'
  AND i_rev.항목명 LIKE '%매출액%'
  AND b_debt.항목명 = '부채총계'
  AND b_equity.항목명 = '자본총계'
  AND (i_op.당기_반기_누적 * 100.0 / i_rev.당기_반기_누적) >= 10  -- 영업이익률 10%+
  AND (b_debt.당기_반기말 * 100.0 / b_equity.당기_반기말) < 50   -- 부채비율 50%-
ORDER BY 영업이익률 DESC
LIMIT 20;

-- "매출액 10조 이상, 순이익 1조 이상 기업"
SELECT DISTINCT
    i_rev.회사명,
    i_rev.당기_반기_누적 as 매출액,
    i_net.당기_반기_누적 as 순이익
FROM income_statement i_rev
JOIN income_statement i_net 
    ON i_rev.회사명 = i_net.회사명 
    AND i_rev.결산기준일 = i_net.결산기준일
WHERE i_rev.항목명 LIKE '%매출액%'
  AND i_rev.당기_반기_누적 > 10000000000000  -- 10조
  AND i_net.항목명 = '반기순이익'
  AND i_net.당기_반기_누적 > 1000000000000   -- 1조
ORDER BY i_rev.당기_반기_누적 DESC
LIMIT 20;
```

**Important Notes for Multiple Conditions:**
1. Always use DISTINCT to avoid duplicate rows
2. JOIN on both 회사명 AND 결산기준일
3. Use meaningful column aliases (as 영업이익, as 자산총계)
4. Add ORDER BY to show most relevant results first
5. Use LIMIT to prevent too many results (default 10-20, or 100 if user asks for "모두")
6. Number formats: 1000억 = 100000000000, 1조 = 1000000000000

**CRITICAL: Calculating Ratios in SQL for Filtering (비율로 필터링)**
When filtering by ratio conditions (예: "영업이익률 20% 이상"):
1. Calculate ratio in SELECT: `ROUND(영업이익 * 100.0 / 매출액, 2) as 영업이익률`
2. Filter using the SAME calculation: `WHERE (영업이익 * 100.0 / 매출액) >= 20`
3. The calculated column will show percentage value (예: 14.05 means 14.05%)
4. Include both raw data AND calculated ratio in SELECT for transparency
5. **CRITICAL**: Use EXACT item name matching to avoid partial matches!
   - 매출액: Use `항목명 = '매출액'` NOT `항목명 LIKE '%매출액%'`
   - 영업수익: Use `항목명 = '영업수익'` NOT `항목명 LIKE '%영업수익%'`
   - 영업이익: Use `(항목명 = '영업이익' OR 항목명 = '영업이익(손실)')` NOT `항목명 LIKE '%영업이익%'`
   - This prevents matching "건설계약으로 인한 매출액" or "재화의 판매로 인한 매출액"

**Example 1: Filter by Operating Profit Margin (단일 조건)**
```sql
-- "영업이익률 20% 이상이면서 매출액 1000억 이상인 회사 모두"
SELECT DISTINCT
    i_op.회사명,
    i_op.당기_반기_누적 as 영업이익,
    i_rev.당기_반기_누적 as 매출액,
    ROUND(CAST(REPLACE(i_op.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL), 2) as 영업이익률
FROM income_statement i_op
JOIN income_statement i_rev 
    ON i_op.회사명 = i_rev.회사명 
    AND i_op.결산기준일 = i_rev.결산기준일
WHERE (i_op.항목명 = '영업이익' OR i_op.항목명 = '영업이익(손실)')
  AND (i_rev.항목명 = '매출액' OR i_rev.항목명 = '영업수익')
  AND CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL) >= 100000000000  -- 1000억 이상
  AND (CAST(REPLACE(i_op.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
       CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL)) >= 20  -- 영업이익률 20%+
ORDER BY 영업이익률 DESC
LIMIT 100;  -- "모두" 조회이므로 100
```

**Example 2: Range Condition (범위 조건) - CRITICAL!**
```sql
-- "매출액이 100억 이상 1000억 미만이면서, 영업이익률이 20% 이상인 기업 모두 추출해줘"
SELECT DISTINCT
    i_op.회사명,
    i_rev.당기_반기_누적 as 매출액,
    i_op.당기_반기_누적 as 영업이익,
    ROUND(CAST(REPLACE(i_op.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL), 2) as 영업이익률
FROM income_statement i_op
JOIN income_statement i_rev 
    ON i_op.회사명 = i_rev.회사명 
    AND i_op.결산기준일 = i_rev.결산기준일
WHERE (i_op.항목명 = '영업이익' OR i_op.항목명 = '영업이익(손실)')
  AND (i_rev.항목명 = '매출액' OR i_rev.항목명 = '영업수익')
  AND CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL) >= 10000000000   -- 100억 이상
  AND CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL) < 100000000000   -- 1000억 미만 (CRITICAL!)
  AND (CAST(REPLACE(i_op.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
       CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL)) >= 20  -- 영업이익률 20%+
ORDER BY 영업이익률 DESC
LIMIT 100;  -- "모두" 추출이므로 100
```

**CRITICAL: Data Type Issue - Numbers are stored as TEXT with commas!**
- Values like "11,361,329,000,000" are stored as TEXT, not REAL
- You MUST remove commas before numeric comparison: `CAST(REPLACE(column, ',', '') AS REAL)`
- For WHERE conditions: `CAST(REPLACE(당기_반기_누적, ',', '') AS REAL) > 100000000000`
- For ORDER BY: `ORDER BY CAST(REPLACE(당기_반기_누적, ',', '') AS REAL) DESC`

**Corrected Example:**
```sql
SELECT DISTINCT
    i.회사명,
    i.당기_반기_누적 as 영업이익,
    b.당기_반기말 as 자산총계
FROM income_statement i
JOIN balance_sheet b 
    ON i.회사명 = b.회사명 
    AND i.결산기준일 = b.결산기준일
WHERE i.항목명 LIKE '%영업이익%'
  AND CAST(REPLACE(i.당기_반기_누적, ',', '') AS REAL) > 100000000000  -- Remove comma!
  AND b.항목명 LIKE '%자산총계%'
  AND CAST(REPLACE(b.당기_반기말, ',', '') AS REAL) > 1000000000000    -- Remove comma!
ORDER BY CAST(REPLACE(i.당기_반기_누적, ',', '') AS REAL) DESC
LIMIT 20;
```

## IMPORTANT: Period/Time-based Data Selection
**상반기 (Half-year) Data:**
- When the question mentions "상반기" (half-year), "2025년 상반기", or "반기" (semi-annual):
  - For income_statement table, use the column `당기_반기_누적` (current half-year accumulated)
  - DO NOT add WHERE conditions like "WHERE period = '상반기'" - no such column exists!
  - The column name itself indicates the period

**Examples:**
- "삼성전자의 상반기 영업이익은?" → SELECT 당기_반기_누적 FROM income_statement WHERE 회사명='삼성전자' AND 항목명 LIKE '%영업이익%'
- "2025년 상반기 매출" → SELECT 당기_반기_누적 FROM income_statement WHERE 항목명 LIKE '%매출%'
- "반기 데이터" → Use 당기_반기_누적 column

**Column meanings:**
- 당기_반기_누적 = Current half-year accumulated (상반기 누적)
- 당기_반기_3개월 = Current half-year 3-month
- 전기_반기_누적 = Previous half-year accumulated
- 전기 = Previous year (전년도)
- 전전기 = Year before previous

## CRITICAL: ROE, ROA, 부채비율 - JOIN balance_sheet!
**When the question asks for ROE, ROA, 부채비율, 유동비율:**
- These ratios require data from BOTH income_statement AND balance_sheet
- You MUST JOIN the two tables!

**Example 3: ROE Calculation (ROE = 순이익 / 자본총계 × 100)**
```sql
-- "SK텔레콤의 매출액, 영업이익, ROE 조회해줘"
SELECT 
    i.회사명,
    i_rev.당기_반기_누적 as 매출액,
    i_op.당기_반기_누적 as 영업이익,
    i.당기_반기_누적 as 순이익,
    b.당기_반기말 as 자본총계,
    ROUND(CAST(REPLACE(i.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(b.당기_반기말, ',', '') AS REAL), 2) as ROE
FROM income_statement i
JOIN balance_sheet b 
    ON i.회사명 = b.회사명 AND i.결산기준일 = b.결산기준일
    AND b.항목명 = '자본총계'
LEFT JOIN income_statement i_rev
    ON i.회사명 = i_rev.회사명 AND i.결산기준일 = i_rev.결산기준일
    AND (i_rev.항목명 = '매출액' OR i_rev.항목명 = '영업수익')
LEFT JOIN income_statement i_op
    ON i.회사명 = i_op.회사명 AND i.결산기준일 = i_op.결산기준일
    AND (i_op.항목명 = '영업이익' OR i_op.항목명 = '영업이익(손실)')
WHERE i.회사명 = 'SK텔레콤'
  AND (i.항목명 = '당기순이익' OR i.항목명 = '반기순이익')
LIMIT {top_k};
```

**Example 4: ROA Calculation (ROA = 순이익 / 자산총계 × 100)**
```sql
-- "삼성전자의 ROA와 부채비율 조회해줘"
SELECT 
    i.회사명,
    i.당기_반기_누적 as 순이익,
    b_asset.당기_반기말 as 자산총계,
    b_equity.당기_반기말 as 자본총계,
    b_debt.당기_반기말 as 부채총계,
    ROUND(CAST(REPLACE(i.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(b_asset.당기_반기말, ',', '') AS REAL), 2) as ROA,
    ROUND(CAST(REPLACE(b_debt.당기_반기말, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(b_equity.당기_반기말, ',', '') AS REAL), 2) as 부채비율
FROM income_statement i
JOIN balance_sheet b_asset 
    ON i.회사명 = b_asset.회사명 AND i.결산기준일 = b_asset.결산기준일
    AND b_asset.항목명 = '자산총계'
JOIN balance_sheet b_equity
    ON i.회사명 = b_equity.회사명 AND i.결산기준일 = b_equity.결산기준일
    AND b_equity.항목명 = '자본총계'
LEFT JOIN balance_sheet b_debt
    ON i.회사명 = b_debt.회사명 AND i.결산기준일 = b_debt.결산기준일
    AND b_debt.항목명 = '부채총계'
WHERE i.회사명 = '삼성전자'
  AND (i.항목명 = '반기순이익' OR i.항목명 = '당기순이익')
LIMIT {top_k};
```

**Example 5: ROE Filtering with Revenue Range (ROE 필터링 + 매출액 범위 조건) - CRITICAL!**
```sql
-- "매출액이 100억에서 1000억 사이인데, ROE가 10% 이상인 기업 리스트 모두 추출해줘"
-- CRITICAL: This requires income_statement + balance_sheet JOIN + multiple conditions!
SELECT 
    i.회사명,
    i_rev.당기_반기_누적 as 매출액,
    i.당기_반기_누적 as 순이익,
    b.당기_반기말 as 자본총계,
    ROUND(CAST(REPLACE(i.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
          CAST(REPLACE(b.당기_반기말, ',', '') AS REAL), 2) as ROE
FROM income_statement i
JOIN balance_sheet b 
    ON i.회사명 = b.회사명 
    AND i.결산기준일 = b.결산기준일
    AND b.항목명 = '자본총계'
LEFT JOIN income_statement i_rev
    ON i.회사명 = i_rev.회사명 
    AND i.결산기준일 = i_rev.결산기준일
    AND (i_rev.항목명 = '매출액' OR i_rev.항목명 = '영업수익')
WHERE (i.항목명 = '당기순이익' OR i.항목명 = '반기순이익')
  AND CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL) >= 10000000000    -- 100억 이상
  AND CAST(REPLACE(i_rev.당기_반기_누적, ',', '') AS REAL) < 100000000000    -- 1000억 미만
  AND (CAST(REPLACE(i.당기_반기_누적, ',', '') AS REAL) * 100.0 / 
       CAST(REPLACE(b.당기_반기말, ',', '') AS REAL)) >= 10                -- ROE 10% 이상
ORDER BY ROE DESC
LIMIT 100;  -- "모두" 추출이므로 100
```

**CRITICAL Notes for ROE/ROA Filtering:**
1. **Always use main table alias consistently**: `FROM income_statement i` → Use `i.회사명` everywhere
2. **JOIN order matters**: First JOIN balance_sheet for 자본총계, then LEFT JOIN for additional data
3. **WHERE clause for ratios**: Calculate ratio in WHERE using same formula as SELECT
4. **Don't forget both range bounds**: `>= 100억 AND < 1000억` (not just `>= 100억`)

**Key Points for balance_sheet JOIN:**
- income_statement uses: `당기_반기_누적` (accumulated)
- balance_sheet uses: `당기_반기말` (end of period)
- JOIN condition: `ON i.회사명 = b.회사명 AND i.결산기준일 = b.결산기준일`
- Always specify `항목명` in JOIN: `AND b.항목명 = '자본총계'`

Question: {input}
""")
        
        def write_query(state: State):
            """SQL 쿼리를 생성합니다 (고유명사 정보 활용)."""
            # 질문에서 고유명사 검색
            entity_info = self.search_entities(state["question"])
            
            prompt = query_prompt_template.invoke({
                "dialect": self.db.dialect,
                "top_k": 10,
                "table_info": self.db.get_table_info(),
                "input": state["question"],
                "entity_info": entity_info if entity_info else "No specific entities found"
            })
            
            structured_llm = self.llm.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            return {"query": result["query"]}
        
        def execute_query(state: State):
            """SQL 쿼리를 실행합니다."""
            execute_query_tool = QuerySQLDatabaseTool(db=self.db)
            return {"result": execute_query_tool.invoke(state["query"])}
        
        def generate_answer(state: State):
            """쿼리 결과를 바탕으로 답변을 생성합니다."""
            prompt = (
                "Given the following user question, corresponding SQL query, "
                "and SQL result, answer the user question in Korean.\n\n"
                f'Question: {state["question"]}\n'
                f'SQL Query: {state["query"]}\n'
                f'SQL Result: {state["result"]}\n\n'
                "**🚨 CRITICAL: ALWAYS show company name (회사명) for EVERY data point! 🚨**\n"
                "- If SQL Result has multiple companies, group data by company name\n"
                "- Format: '**[Company Name]**: 매출액: X, 영업이익: Y, ROE: Z%'\n"
                "- NEVER show numbers without company names - users can't tell which data belongs to which company!\n"
                "- If multiple rows exist, organize by company first\n\n"
                "**Example (GOOD):**\n"
                "- '**SK텔레콤**: 매출액: 8조원, 영업이익: 9,056억원, ROE: 3.72%'\n"
                "- '**SK하이닉스**: 매출액: 39조원, 영업이익: 16조원, ROE: 5.86%'\n\n"
                "**Example (BAD - DO NOT DO THIS!):**\n"
                "- '매출액: 8조원, 영업이익: 9,056억원' (회사명 없음 ❌)\n\n"
                "**CRITICAL: Financial Ratio - Check if Already Calculated in SQL!**\n"
                "1. First, check if SQL Result already has ratio columns (영업이익률, 순이익률, ROE, etc.)\n"
                "2. If YES: Use the calculated value AS IS (already in percentage) - DO NOT recalculate!\n"
                "3. If NO: Calculate it yourself from the raw data\n\n"
                "**Example:**\n"
                "- SQL Result has '영업이익률: 14.05' → Answer: '영업이익률: 14.05%' (just add %)\n"
                "- SQL Result has '영업이익: 1000, 매출액: 5000' → Calculate: '영업이익률: 20% (1000÷5000×100)'\n\n"
                "**CRITICAL: Financial Ratio Calculation (재무비율 자동 계산)**\n"
                "If the question asks for ratios (영업이익률, 순이익률, ROE, ROA, 부채비율, etc.),\n"
                "and SQL Result contains the necessary data, YOU MUST CALCULATE IT!\n\n"
                "**🚨 CRITICAL: Do NOT confuse 영업수익 vs 영업이익! 🚨**\n"
                "- 영업수익 (Operating Revenue) = 매출액 (Revenue) = Total sales/income\n"
                "- 영업이익 (Operating Profit/Income) = 영업수익 - 영업비용 = Profit after costs\n"
                "- **NEVER say '영업수익 = 영업이익'! They are COMPLETELY DIFFERENT!**\n"
                "- If SQL Result only has '영업수익' but NOT '영업이익', you MUST say:\n"
                "  '영업수익은 X원입니다. 영업이익 정보는 제공되지 않았습니다.'\n\n"
                "**Common Ratios:**\n"
                "1. 영업이익률 (Operating Profit Margin) = (영업이익 / 매출액 or 영업수익) × 100\n"
                "   ⚠️ Numerator MUST be 영업이익, NOT 영업수익!\n"
                "2. 순이익률 (Net Profit Margin) = (순이익 / 매출액 or 영업수익) × 100\n"
                "3. ROE (자기자본이익률) = (순이익 / 자본총계) × 100\n"
                "4. ROA (총자산이익률) = (순이익 / 자산총계) × 100\n"
                "5. 부채비율 (Debt Ratio) = (부채총계 / 자본총계) × 100\n\n"
                "**How to Calculate:**\n"
                "1. Remove commas from numbers: '47,687,046,619' → 47687046619\n"
                "2. Divide and multiply by 100 for percentage\n"
                "3. Round to 2 decimal places\n"
                "4. Show calculation in answer: '영업이익률 = (47,289,352,211 ÷ 336,666,812,235) × 100 = 14.05%'\n\n"
                "**Example:**\n"
                "Question: 'SNT다이내믹스의 매출액과 영업이익, 영업이익률 조회해줘'\n"
                "SQL Result: 매출액: 336,666,812,235, 영업이익: 47,289,352,211\n"
                "Answer: '매출액: 336,666,812,235원, 영업이익: 47,289,352,211원, 영업이익률: 14.05% (계산: 47,289,352,211 ÷ 336,666,812,235 × 100)'\n\n"
                "**NEVER say '영업이익률에 대한 정보는 제공되지 않았습니다' if you can calculate it!**\n\n"
                "**CRITICAL: Number Formatting Rules**\n"
                "- **NEVER calculate or convert number units yourself - you make mistakes!**\n"
                "- **Use the EXACT numbers from SQL Result with commas (e.g., 47,687,046,619원)**\n"
                "- **DO NOT convert to 억, 조, 만 units - just use the original number!**\n"
                "- If you must provide a readable format, keep the original: '47,687,046,619원'\n"
                "- Example: '매출액은 47,687,046,619원입니다' (NOT '476억원' or '4,768억원')\n\n"
                "Other Important Rules:\n"
                "- If the SQL result contains data, provide the specific numbers/values in your answer\n"
                "- When mentioning '상반기' (half-year) data, explain it's the accumulated data for the first half\n"
                "- Be specific and concrete based on the actual SQL result\n"
                "- Keep all numbers exactly as they appear in SQL Result (with commas)\n\n"
                "Bad Examples (DO NOT DO THIS):\n"
                "- ❌ '매출액은 4,768억 7,046만원' (wrong conversion!)\n"
                "- ❌ '영업이익은 867억원' (wrong conversion!)\n"
                "- ❌ '영업이익률에 대한 정보는 제공되지 않았습니다' (when you can calculate it!)\n\n"
                "Good Examples:\n"
                "- ✅ '매출액은 47,687,046,619원입니다'\n"
                "- ✅ '영업이익은 8,675,711,602원입니다'\n"
                "- ✅ '순이익은 6,588,565,249원입니다'\n"
                "- ✅ '영업이익률은 18.22%입니다 (계산: 8,675,711,602 ÷ 47,687,046,619 × 100)'"
            )
            response = self.llm.invoke(prompt)
            return {"answer": response.content}
        
        # StateGraph 생성
        graph_builder = StateGraph(State).add_sequence(
            [write_query, execute_query, generate_answer]
        )
        graph_builder.add_edge(START, "write_query")
        
        return graph_builder.compile()
    
    def query_financial_data(self, question: str) -> str:
        """재무 데이터를 조회합니다 (Text2SQL)."""
        try:
            result = self.text2sql_graph.invoke({"question": question})
            return result.get("answer", "답변을 생성할 수 없습니다.")
        except Exception as e:
            return f"재무 데이터 조회 중 오류가 발생했습니다: {str(e)}"
    
    def search_web(self, query: str) -> str:
        """웹 검색을 수행합니다."""
        try:
            search_result = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            
            if not search_result.get("results"):
                return "검색 결과를 찾을 수 없습니다."
            
            # 검색 결과를 포맷팅
            formatted_results = []
            for result in search_result["results"][:3]:
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                
                formatted_results.append(f"제목: {title}\n내용: {content}\n출처: {url}\n")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"웹 검색 중 오류가 발생했습니다: {str(e)}"


# 전역 도구 인스턴스 (캐싱용)
_tools_instance = None


def get_tools_instance(force_reload=False):
    """도구 인스턴스를 반환합니다.
    
    Args:
        force_reload: True이면 기존 인스턴스를 무시하고 새로 생성
    """
    global _tools_instance
    if _tools_instance is None or force_reload:
        _tools_instance = FinancialAnalysisTools()
    return _tools_instance

