import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tools import get_tools_instance

# 환경 변수 로드
load_dotenv()


class FinancialAnalysisState(TypedDict):
    """재무제표 분석 시스템의 상태를 정의합니다."""
    messages: List[BaseMessage]
    route_decision: str
    current_query: str
    intermediate_results: List[str]
    final_answer: str
    iteration_count: int


class FinancialAnalysisGraph:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.openai_api_key
        )
        
        # 도구 인스턴스
        self.tools_instance = get_tools_instance()
        
        # 메모리 설정
        self.memory = MemorySaver()
        
        # 그래프 빌드
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Adaptive RAG 워크플로우를 구성합니다."""
        
        # StateGraph 생성
        workflow = StateGraph(FinancialAnalysisState)
        
        # 노드 추가
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("no_retrieval", self.no_retrieval_node)
        workflow.add_node("single_shot_rag", self.single_shot_rag_node)
        workflow.add_node("iterative_rag", self.iterative_rag_node)
        workflow.add_node("generate_response", self.generate_response_node)
        
        # 시작점 설정
        workflow.set_entry_point("analyze_query")
        
        # 조건부 엣지 설정 (라우팅 로직)
        workflow.add_conditional_edges(
            "analyze_query",
            self.route_decision_function,
            {
                "no_retrieval": "no_retrieval",
                "single_shot_rag": "single_shot_rag", 
                "iterative_rag": "iterative_rag"
            }
        )
        
        # 각 RAG 노드에서 응답 생성으로
        workflow.add_edge("no_retrieval", "generate_response")
        workflow.add_edge("single_shot_rag", "generate_response")
        
        # iterative_rag는 조건부로 자기 자신 또는 generate_response로
        workflow.add_conditional_edges(
            "iterative_rag",
            self.should_continue_iteration,
            {
                "continue": "iterative_rag",  # 계속 반복
                "finish": "generate_response"  # 완료
            }
        )
        
        # 응답 생성 후 종료
        workflow.add_edge("generate_response", END)
        
        # 메모리와 함께 컴파일
        return workflow.compile(checkpointer=self.memory)
    
    def analyze_query_node(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """사용자 질문을 분석하여 라우팅 결정을 내립니다."""
        
        user_message = state["messages"][-1].content
        
        # 대화 컨텍스트 구성 (최근 3개 메시지)
        recent_messages = state["messages"][-3:] if len(state["messages"]) > 1 else state["messages"]
        conversation_context = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Bot'}: {msg.content[:100]}..."
            for msg in recent_messages[:-1]  # 마지막 메시지 제외
        ]) if len(recent_messages) > 1 else ""
        
        # 대화 기록 포맷팅 (f-string 밖에서 처리)
        context_section = f"최근 대화 기록:\n{conversation_context}\n\n" if conversation_context else ""
        
        analysis_prompt = f"""
다음 사용자 질문을 분석하여 적절한 처리 방법을 결정해주세요:

{context_section}현재 질문: "{user_message}"

        다음 중 하나를 선택해주세요:
        
        1. "no_retrieval": 
           - 일반적인 상식 질문 (예: "재무제표가 뭐야?", "손익계산서란?")
           - 단순한 정의나 설명 요청
           - LLM의 자체 지식으로 충분히 답변 가능한 질문
        
        2. "single_shot_rag":
           - 특정 회사의 특정 재무 데이터 조회 (예: "삼성전자 2025년 매출액")
           - 단순한 웹 검색 질문 (예: "최근 AI 트렌드")
           - 한 번의 도구 호출로 답변 가능한 질문
           - **복합 조건이지만 SQL JOIN으로 한 번에 처리 가능** (예: "영업이익 1000억 넘고 자산 1조 이상 기업")
        
        3. "iterative_rag":
           - **복잡한 비교 분석 (예: "삼성전자와 SK하이닉스 매출 비교하고 그 차이 원인 분석")**
           - **여러 회사의 재무 데이터를  비교하는 질문 (예: "삼성전자와 SK하이닉스의 재무 구조 비교")**
           - **"원인", "이유", "배경" 분석 질문 (예: "SK하이닉스 영업이익 상승의 원인")**
           - **"검색해줘", "찾아줘" 키워드가 있는 질문 (예: "삼성전자 최근 뉴스 검색해줘")**
           - 여러 단계의 계산이나 분석이 필요한 질문
           - 여러 도구를 순차적으로 사용해야 하는 복합 질문
           - **데이터 조회 후 추가 분석/해석이 필요한 질문**
        
        **Important Rules:**
        1. Multiple conditions can often be handled by single_shot_rag with complex SQL JOIN
        2. **"비교", "비교 분석", "compare" 키워드 + 2개 이상 회사명 → MUST use iterative_rag**
        3. **"A와 B의 X, Y, Z 비교" → MUST use iterative_rag** (여러 지표 비교)
        4. **"원인", "이유", "배경", "검색해줘" 키워드 → MUST use iterative_rag** (DB + 웹 검색 필요)
        5. Only use iterative_rag if analysis/interpretation is needed after data retrieval
        
        답변은 반드시 "no_retrieval", "single_shot_rag", "iterative_rag" 중 하나만 출력해주세요.
"""
        
        response = self.llm.invoke(analysis_prompt)
        route_decision = response.content.strip().lower()
        
        print(f"\n[DEBUG] analyze_query_node:")
        print(f"  - 질문: {user_message[:50]}...")
        print(f"  - 원본 LLM 응답: {response.content.strip()}")
        print(f"  - 추출된 route_decision: {route_decision}")
        
        # 유효하지 않은 결정이면 기본값으로 single_shot_rag 사용
        if route_decision not in ["no_retrieval", "single_shot_rag", "iterative_rag"]:
            print(f"  - 유효하지 않은 결정, single_shot_rag로 기본 설정")
            route_decision = "single_shot_rag"
        
        print(f"  → 최종 라우팅: {route_decision}")
        
        return {
            **state,
            "route_decision": route_decision,
            "current_query": user_message,
            "iteration_count": 0
        }
    
    def no_retrieval_node(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """LLM의 자체 지식으로 직접 답변합니다."""
        
        user_message = state["current_query"]
        
        prompt = f"""
다음 질문에 대해 재무/회계 전문 지식을 바탕으로 친절하고 정확하게 답변해주세요:

질문: {user_message}

답변 시 다음 사항을 고려해주세요:
- 한국어로 답변
- 구체적이고 이해하기 쉽게 설명
- 필요시 예시를 포함
- 전문 용어는 간단히 설명
"""
        
        response = self.llm.invoke(prompt)
        
        return {
            **state,
            "final_answer": response.content,
            "intermediate_results": [response.content]
        }
    
    def single_shot_rag_node(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """한 번의 도구 호출로 답변을 생성합니다."""
        
        user_message = state["current_query"]
        
        # 대화 컨텍스트 구성
        if len(state["messages"]) > 1:
            recent_messages = state["messages"][-3:]
            conversation_context = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Bot'}: {msg.content[:150]}"
                for msg in recent_messages[:-1]
            ])
            full_query = f"대화 기록:\n{conversation_context}\n\n현재 질문: {user_message}"
        else:
            full_query = user_message
        
        # 재무 질문인지 판단 (full_query 사용)
        is_financial = self._is_financial_query(full_query)
        
        if is_financial:
            # Text2SQL로 재무 데이터 조회 (full_query 사용)
            tool_result = self.tools_instance.query_financial_data(full_query)
        else:
            # 웹 검색 (full_query 사용)
            tool_result = self.tools_instance.search_web(full_query)
        
        return {
            **state,
            "final_answer": tool_result,
            "intermediate_results": [tool_result]
        }
    
    def iterative_rag_node(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """여러 도구를 반복적으로 사용하여 복잡한 질문에 답변합니다."""
        
        user_message = state["current_query"]
        
        # 대화 컨텍스트 구성 (짧은 질문 보완용)
        if len(state["messages"]) > 1:
            recent_messages = state["messages"][-3:]
            conversation_context = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Bot'}: {msg.content[:150]}"
                for msg in recent_messages[:-1]
            ])
            full_context = f"대화 기록:\n{conversation_context}\n\n현재 질문: {user_message}"
        else:
            full_context = user_message
        
        max_iterations = 3
        current_iteration = state.get("iteration_count", 0)
        
        intermediate_results = state.get("intermediate_results", [])
        
        print(f"\n[DEBUG] iterative_rag_node 실행:")
        print(f"  - current_iteration: {current_iteration}/{max_iterations}")
        print(f"  - intermediate_results 개수: {len(intermediate_results)}")
        
        if current_iteration >= max_iterations:
            # 최대 반복 횟수에 도달하면 최종 답변 생성 (final_answer 설정)
            print(f"  → 최대 반복 횟수 도달, 최종 답변 생성")
            final_state = self._generate_final_answer_from_results(state)
            return final_state
        
        # 질문에서 언급된 회사 추출 (대화 컨텍스트 포함)
        mentioned_companies = []
        full_context_lower = full_context.lower()
        if "삼성전자" in full_context_lower or "삼성" in full_context_lower:
            mentioned_companies.append("삼성전자")
        if "sk하이닉스" in full_context_lower or "하이닉스" in full_context_lower:
            mentioned_companies.append("SK하이닉스")
        if "sk텔레콤" in full_context_lower or "skt" in full_context_lower:
            mentioned_companies.append("SK텔레콤")
        if "케이티" in full_context_lower or "kt" in full_context_lower:
            mentioned_companies.append("케이티")
        if "lg전자" in full_context_lower:
            mentioned_companies.append("LG전자")
        if "lg유플러스" in full_context_lower or "유플러스" in full_context_lower:
            mentioned_companies.append("LG유플러스")
        
        # 이미 조회한 회사 추출 (중복 방지)
        queried_companies = []
        for result in intermediate_results:
            if "삼성전자" in result:
                queried_companies.append("삼성전자")
            elif "SK하이닉스" in result or "sk하이닉스" in result or "하이닉스" in result:
                queried_companies.append("SK하이닉스")
            elif "SK텔레콤" in result or "sk텔레콤" in result:
                queried_companies.append("SK텔레콤")
            elif "케이티" in result or "KT" in result:
                queried_companies.append("케이티")
        queried_companies = list(set(queried_companies))  # 중복 제거
        
        # 아직 조회하지 않은 회사
        remaining_companies = [c for c in mentioned_companies if c not in queried_companies]
        
        print(f"  - 질문에서 언급된 회사: {mentioned_companies}")
        print(f"  - 이미 조회한 회사: {queried_companies}")
        print(f"  - 아직 조회하지 않은 회사: {remaining_companies}")
        
        # 현재 상황 분석 및 다음 도구 선택
        analysis_prompt = f"""
다음 복잡한 질문을 단계별로 분석하고 해결하기 위한 다음 단계를 결정해주세요:

{full_context}

**질문에서 언급된 회사: {', '.join(mentioned_companies) if mentioned_companies else "없음"}**
**이미 조회한 회사: {', '.join(queried_companies) if queried_companies else "없음"}**
**🔴 아직 조회하지 않은 회사 (반드시 먼저 조회!): {', '.join(remaining_companies) if remaining_companies else "없음 - 모두 조회 완료"}**

현재까지의 결과:
{chr(10).join(intermediate_results) if intermediate_results else "아직 결과 없음"}

현재 반복 횟수: {current_iteration + 1}/{max_iterations}

**CRITICAL RULES:**
1. **"아직 조회하지 않은 회사"가 있으면 반드시 그 회사를 먼저 조회하세요!**
   - 위에 🔴로 표시된 회사가 있으면 **반드시 그 회사를 조회**해야 합니다
   - 이미 조회한 회사를 다시 조회하면 안 됩니다!
   - 예: 아직 조회하지 않은 회사: 삼성전자
     → 반드시: "선택: financial_query | 쿼리: 삼성전자 매출액, 영업이익, 순이익"
   
2. 재무 데이터(매출액, 영업이익, 순이익, 자산 등) 관련 질문은 **반드시 먼저 financial_query로 DB 조회**

3. **"비교 분석" 질문의 경우 - 매우 중요!:**
   - 질문에 언급된 **모든 회사**의 데이터를 조회해야 함
   - **아직 조회하지 않은 회사를 우선 조회할 것!**
   - 모든 회사 데이터가 수집되면 final_answer

4. **"원인", "이유", "배경" 질문의 경우:**
   - 먼저 관련 재무 데이터 조회 (financial_query)
   - 그 다음 웹에서 원인/이유 검색 (web_search) - **필수!**
   - 두 정보를 종합하여 final_answer

5. **"검색해줘", "찾아줘" 키워드가 있으면 반드시 web_search 사용!**

6. **산업별 항목명 차이 - 매우 중요!:**
   - **제조업 (삼성전자, SK하이닉스 등)**: "매출액" 사용
   - **금융/통신업 (SK텔레콤, 케이티, LG유플러스 등)**: "영업수익" 사용 (매출액 대신!)
   - **순이익**: 현재 반기 데이터이므로 "반기순이익" 사용 (당기순이익 아님!)
   - **쿼리 작성 시 반드시 두 가지 패턴 모두 포함:**
     - 매출: "매출액, 영업수익 조회"
     - 순이익: "반기순이익, 순이익 조회"

7. **재무 비율(영업이익률, 순이익률 등) 질문:**
   - 비율 컬럼은 DB에 없음! 직접 계산해야 함
   - 영업이익률 = 영업이익 / 매출액 (or 영업수익)
   - 순이익률 = 순이익 / 매출액 (or 영업수익)
   - **쿼리 예: "삼성전자 매출액, 영업이익, 순이익" (비율은 빼고!)**
   - **중요: 비율 컬럼을 직접 조회하지 말고, 기본 데이터만 조회!**

7. **절대로 LLM의 자체 지식으로 재무 데이터를 추정하지 마세요!**

8. intermediate_results가 비어있으면 final_answer 선택 금지!

다음 중 하나를 선택해주세요:
1. "financial_query": 재무 데이터베이스에서 추가 정보 조회 (재무 데이터 필수!)
2. "web_search": 웹에서 추가 정보 검색 (원인/이유/배경/최신 뉴스 필수!)
3. "final_answer": 충분한 정보가 모였으므로 최종 답변 생성 (데이터가 있을 때만!)

**CRITICAL: One Step at a Time (한 번에 하나씩!)**
- **반드시 한 번에 하나의 선택만 하세요!**
- **여러 회사를 비교할 때도 한 번에 한 회사씩 조회하세요!**
- **절대로 한 번에 여러 "선택:"을 작성하지 마세요!**

**비교 분석 질문 예시:**
- "삼성전자와 SK하이닉스 매출액, 영업이익, 순이익 비교" 
  → Step 1: "선택: financial_query | 쿼리: 삼성전자 매출액, 영업이익, 반기순이익"
  → Step 2: "선택: financial_query | 쿼리: SK하이닉스 매출액, 영업이익, 반기순이익"
  → Step 3: "선택: final_answer" (비율은 답변 생성 시 계산)

- "SK텔레콤, 케이티, LG유플러스 영업이익률 비교" (통신사 - 영업수익 사용!)
  → Step 1: "선택: financial_query | 쿼리: SK텔레콤 영업수익, 영업이익"
  → Step 2: "선택: financial_query | 쿼리: 케이티 영업수익, 영업이익"
  → Step 3: "선택: financial_query | 쿼리: LG유플러스 영업수익, 영업이익"
  → Step 4: "선택: final_answer" (영업이익률 = 영업이익/영업수익 계산)

**잘못된 예시 (하지 마세요!):**
❌ "선택: financial_query | 쿼리: 삼성전자... \n선택: financial_query | 쿼리: SK하이닉스..."
   (한 번에 두 개 선택 - 금지!)
❌ "쿼리: 케이티 매출액" → 케이티는 통신사이므로 "영업수익" 사용!
❌ "쿼리: 삼성전자 당기순이익" → 현재는 반기 데이터이므로 "반기순이익" 사용!

**올바른 예시:**
✅ "선택: financial_query | 쿼리: 삼성전자 매출액, 영업이익, 순이익"
   (한 번에 하나만!)

**원인/이유 분석 질문 예시:**
- "SK하이닉스 영업이익 상승의 원인에 대해서 검색해줘"
  → Step 1: "선택: financial_query | 쿼리: SK하이닉스 영업이익, 매출액"
  → Step 2: "선택: web_search | 쿼리: SK하이닉스 영업이익 상승 원인 2025" - **필수!**
  → Step 3: "선택: final_answer"

- "삼성전자 매출 감소 이유는?"
  → Step 1: "선택: financial_query | 쿼리: 삼성전자 매출액"
  → Step 2: "선택: web_search | 쿼리: 삼성전자 매출 감소 원인" - **필수!**
  → Step 3: "선택: final_answer"

**중요: 영업이익률, 순이익률 등 비율은 DB에 없으므로 쿼리에서 제외하고, 매출액과 영업이익(또는 순이익)만 조회하세요!**

선택과 함께 구체적인 쿼리도 함께 제시해주세요.
형식: "선택: [선택값] | 쿼리: [구체적인 쿼리]"
"""
        
        response = self.llm.invoke(analysis_prompt)
        decision_text_full = response.content.strip()
        
        print(f"  - LLM 결정 (전체): {decision_text_full}")
        
        # 첫 번째 줄만 처리 (여러 줄이 있을 경우 대비)
        decision_text = decision_text_full.split("\n")[0].strip()
        if decision_text != decision_text_full:
            print(f"  - 여러 줄 감지, 첫 번째 줄만 사용: {decision_text}")
        
        # 결정 파싱
        if "선택: financial_query" in decision_text:
            query_part = decision_text.split("쿼리: ")[-1] if "쿼리: " in decision_text else full_context
            print(f"  → financial_query 실행: {query_part[:50]}...")
            tool_result = self.tools_instance.query_financial_data(query_part)
            intermediate_results.append(f"반복 {current_iteration + 1}: {tool_result}")
            
            # 다음 반복이 최대 횟수에 도달하면 바로 final_answer 생성
            if current_iteration + 1 >= max_iterations:
                print(f"  → 다음 반복이 최대 횟수 도달 예정, 최종 답변 생성")
                updated_state = {
                    **state,
                    "iteration_count": current_iteration + 1,
                    "intermediate_results": intermediate_results
                }
                final_state = self._generate_final_answer_from_results(updated_state)
                print(f"  → final_answer 생성 완료: {len(final_state.get('final_answer', ''))} 글자")
                return final_state
            
            return {
                **state,
                "iteration_count": current_iteration + 1,
                "intermediate_results": intermediate_results
            }
        elif "선택: web_search" in decision_text:
            query_part = decision_text.split("쿼리: ")[-1] if "쿼리: " in decision_text else full_context
            print(f"  → web_search 실행: {query_part[:50]}...")
            tool_result = self.tools_instance.search_web(query_part)
            intermediate_results.append(f"반복 {current_iteration + 1}: {tool_result}")
            
            # 다음 반복이 최대 횟수에 도달하면 바로 final_answer 생성
            if current_iteration + 1 >= max_iterations:
                print(f"  → 다음 반복이 최대 횟수 도달 예정, 최종 답변 생성")
                updated_state = {
                    **state,
                    "iteration_count": current_iteration + 1,
                    "intermediate_results": intermediate_results
                }
                final_state = self._generate_final_answer_from_results(updated_state)
                print(f"  → final_answer 생성 완료: {len(final_state.get('final_answer', ''))} 글자")
                return final_state
            
            return {
                **state,
                "iteration_count": current_iteration + 1,
                "intermediate_results": intermediate_results
            }
        else:
            # 최종 답변 생성으로 진행 (final_answer 설정)
            print(f"  → final_answer 선택, 최종 답변 생성")
            final_state = self._generate_final_answer_from_results(state)
            print(f"  → final_answer 생성 완료: {len(final_state.get('final_answer', ''))} 글자")
            return final_state
    
    def _generate_final_answer_from_results(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """수집된 결과들을 바탕으로 최종 답변을 생성합니다."""
        
        user_message = state["current_query"]
        
        # 대화 컨텍스트 구성 (원래 질문의 의도 파악용)
        if len(state["messages"]) > 1:
            recent_messages = state["messages"][-3:]
            conversation_context = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Bot'}: {msg.content[:150]}"
                for msg in recent_messages[:-1]
            ])
            full_question = f"대화 기록:\n{conversation_context}\n\n현재 질문: {user_message}"
        else:
            full_question = user_message
        
        intermediate_results = state.get("intermediate_results", [])
        
        print(f"\n[DEBUG] _generate_final_answer_from_results:")
        print(f"  - intermediate_results 개수: {len(intermediate_results)}")
        for i, result in enumerate(intermediate_results):
            print(f"  - 결과 {i+1}: {result[:200]}...")
        
        # 재무 데이터 관련 질문인데 결과가 없으면 경고
        if not intermediate_results:
            return {
                **state,
                "response": "죄송합니다. 재무 데이터를 조회하지 못했습니다. 다시 질문해주시면 데이터베이스에서 정확한 정보를 조회하여 답변드리겠습니다.",
                "route_decision": "error_no_data"
            }
        
        final_prompt = f"""
다음 복잡한 질문에 대해 수집된 모든 정보를 종합하여 완전하고 정확한 답변을 제공해주세요:

{full_question}

수집된 정보:
{chr(10).join(intermediate_results)}

**CRITICAL: 답변 규칙**
- **반드시 수집된 정보만 사용하세요!**
- **절대로 LLM의 자체 지식이나 추정치를 사용하지 마세요!**
- 수집된 정보에 없는 데이터는 "데이터 없음"으로 표시
- "약", "대략", "추정" 같은 표현 금지 (정확한 숫자만 사용)
- 비교 분석 시 반드시 실제 조회된 데이터 기반으로만 작성
- **LaTeX 수식 사용 금지!** (\\text, \\frac 같은 수식 표현 절대 금지)

답변 시 다음 사항을 고려해주세요:
- 한국어로 답변
- 모든 관련 정보를 종합하여 포괄적인 답변 제공
- 비교 분석이 필요한 경우 명확한 비교 표시
- 결론과 인사이트 포함
- 이해하기 쉽게 구조화된 답변
- 모든 숫자는 조회된 정확한 값만 사용 (추정 금지!)
- **비율 계산 시 일반 텍스트로 작성** (예: "순이익률 = (13,339,313,000,000 / 153,706,820,000,000) × 100 = 8.68%")
- **절대로 LaTeX 수식 형식 사용하지 말 것!**

**잘못된 예시 (하지 마세요!):**
❌ LaTeX 형식 사용 금지!

**올바른 예시:**
✅ "순이익률 = (13,339,313,000,000 / 153,706,820,000,000) × 100 = 8.68%"
✅ "순이익률: 8.68% (계산: 순이익 13조 ÷ 매출 154조)"
"""
        
        response = self.llm.invoke(final_prompt)
        
        return {
            **state,
            "final_answer": response.content
        }
    
    def generate_response_node(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """최종 응답을 생성하고 대화 기록에 추가합니다."""
        
        final_answer = state["final_answer"]
        
        print(f"\n[DEBUG] generate_response_node:")
        print(f"  - final_answer 길이: {len(final_answer)} 글자")
        print(f"  - final_answer 내용 (처음 100자): {final_answer[:100] if final_answer else '(비어있음)'}...")
        
        # AI 메시지 추가
        ai_message = AIMessage(content=final_answer)
        state["messages"].append(ai_message)
        
        return state
    
    def route_decision_function(self, state: FinancialAnalysisState) -> str:
        """라우팅 결정 함수"""
        return state["route_decision"]
    
    def should_continue_iteration(self, state: FinancialAnalysisState) -> str:
        """iterative_rag가 계속 반복될지 결정합니다."""
        
        current_iteration = state.get("iteration_count", 0)
        final_answer = state.get("final_answer", "")
        
        print(f"\n[DEBUG] should_continue_iteration 체크:")
        print(f"  - iteration_count: {current_iteration}")
        print(f"  - final_answer 존재: {bool(final_answer)}")
        print(f"  - intermediate_results 개수: {len(state.get('intermediate_results', []))}")
        
        # final_answer가 이미 설정되었으면 완료
        if final_answer and final_answer != "":
            print(f"  → 결정: finish (final_answer 설정됨)")
            return "finish"
        
        # 최대 반복 횟수에 도달했으면 완료
        max_iterations = 3
        if current_iteration >= max_iterations:
            print(f"  → 결정: finish (최대 반복 횟수 도달)")
            return "finish"
        
        # 계속 반복
        print(f"  → 결정: continue")
        return "continue"
    
    def _is_financial_query(self, query: str) -> bool:
        """질문이 재무 관련인지 판단합니다."""
        financial_keywords = [
            "매출", "매출액", "영업이익", "당기순이익", "자산", "부채", "자본",
            "현금흐름", "재무상태표", "손익계산서", "현금흐름표", "자본변동표",
            "유동자산", "비유동자산", "유동부채", "비유동부채", "이익률", "수익률",
            "회사", "기업", "2023", "2024", "2025", "상반기", "하반기", "분기"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in financial_keywords)
    
    def invoke(self, message: str, config: dict = None) -> str:
        """그래프를 실행하고 결과를 반환합니다."""
        
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # 기존 상태 가져오기 (있으면)
        try:
            existing_state = self.graph.get_state(config)
            
            if existing_state and existing_state.values and "messages" in existing_state.values:
                # 기존 대화에 새 메시지 추가
                updated_state = existing_state.values.copy()
                updated_state["messages"].append(HumanMessage(content=message))
                updated_state["current_query"] = message
                updated_state["route_decision"] = ""
                updated_state["iteration_count"] = 0
                updated_state["intermediate_results"] = []  # 🔥 초기화 필수!
                updated_state["final_answer"] = ""  # 🔥 초기화 필수!
            else:
                # 새로운 대화 시작
                updated_state = {
                    "messages": [HumanMessage(content=message)],
                    "route_decision": "",
                    "current_query": message,
                    "intermediate_results": [],
                    "final_answer": "",
                    "iteration_count": 0
                }
        except Exception as e:
            print(f"상태 조회 실패, 새로운 대화 시작: {e}")
            # 새로운 대화 시작
            updated_state = {
                "messages": [HumanMessage(content=message)],
                "route_decision": "",
                "current_query": message,
                "intermediate_results": [],
                "final_answer": "",
                "iteration_count": 0
            }
        
        # 그래프 실행
        print(f"\n[DEBUG] 그래프 실행 시작")
        result = self.graph.invoke(updated_state, config)
        
        print(f"\n[DEBUG] 그래프 실행 완료")
        print(f"  - final_answer 존재: {bool(result.get('final_answer'))}")
        print(f"  - final_answer 길이: {len(result.get('final_answer', ''))} 글자")
        
        return result["final_answer"]


# 전역 그래프 인스턴스
def get_graph_instance():
    """그래프 인스턴스를 반환합니다."""
    return FinancialAnalysisGraph()

