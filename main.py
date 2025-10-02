import gradio as gr
import os
from dotenv import load_dotenv
from parser import FinancialDataParser
from graph import get_graph_instance

# 환경 변수 로드
load_dotenv()


class FinancialAnalysisApp:
    def __init__(self):
        """재무제표 분석 애플리케이션을 초기화합니다."""
        
        # API 키 확인
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        # 데이터 파서 초기화
        self.parser = FinancialDataParser()
        
        # 그래프는 나중에 초기화 (데이터 로드 후)
        self.graph = None
        
        print("재무제표 분석 시스템이 초기화되었습니다.")
    
    def initialize_data(self):
        """데이터베이스를 초기화하고 재무제표 데이터를 파싱합니다."""
        try:
            print("데이터베이스 초기화 중...")
            self.parser.parse_all_financial_statements()
            print("데이터 초기화가 완료되었습니다.")
            
            # 데이터 로드 후 그래프 초기화 (벡터스토어 빌드)
            print("\n그래프 및 벡터스토어 초기화 중...")
            # force_reload=True로 tools 인스턴스를 새로 생성하여 벡터스토어 재빌드
            from tools import get_tools_instance
            get_tools_instance(force_reload=True)
            self.graph = get_graph_instance()
            print("그래프 초기화가 완료되었습니다.")
            
            return True
        except Exception as e:
            print(f"데이터 초기화 중 오류 발생: {e}")
            return False
    
    def chat_with_system(self, message: str, history: list) -> tuple:
        """시스템과 대화하는 함수"""
        
        if not message.strip():
            return history, ""
        
        # 그래프가 초기화되지 않았으면 오류 메시지 반환
        if self.graph is None:
            error_message = "시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."
            history.append([message, error_message])
            return history, ""
        
        try:
            # 대화 히스토리를 유지하기 위해 동일한 thread_id 사용
            config = {"configurable": {"thread_id": "user_session"}}
            
            # LangGraph를 통해 응답 생성
            response = self.graph.invoke(message, config)
            
            # 대화 기록 업데이트
            history.append([message, response])
            
            return history, ""
            
        except Exception as e:
            error_message = f"오류가 발생했습니다: {str(e)}"
            history.append([message, error_message])
            return history, ""
    
    def create_interface(self):
        """Gradio 인터페이스를 생성합니다."""
        
        # CSS 스타일링 - 더 넓고 모던한 디자인
        css = """
        .gradio-container {
            max-width: 1600px !important;
            margin: auto !important;
            padding: 20px !important;
        }
        
        /* 채팅 메시지 스타일 개선 */
        .message-wrap {
            padding: 15px 20px !important;
            border-radius: 12px !important;
        }
        
        /* 사용자 메시지 */
        .message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        
        /* 봇 메시지 */
        .message.bot {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        }
        
        /* 헤더 스타일 */
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            text-align: center;
            margin-bottom: 20px;
        }
        
        /* 채팅창 스타일 */
        .chatbot {
            border-radius: 15px !important;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1) !important;
        }
        
        /* 입력창 스타일 */
        .input-wrap {
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07) !important;
        }
        
        /* 버튼 스타일 */
        .primary-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 25px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .primary-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(102,126,234,0.4) !important;
        }
        
        /* 예시 버튼 스타일 */
        .example-btn {
            border-radius: 20px !important;
            border: 2px solid #667eea !important;
            color: #667eea !important;
            transition: all 0.3s ease !important;
        }
        
        .example-btn:hover {
            background: #667eea !important;
            color: white !important;
        }
        
        /* 정보 박스 스타일 */
        .info-box {
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }
        """
        
        with gr.Blocks(css=css, title="재무제표 분석 시스템 v2", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 📊 재무제표 분석 AI 시스템 v2
            ### 🤖 LangGraph + GPT-4 기반 지능형 재무 분석 에이전트
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div class="info-box">
                    
                    ## 🚀 주요 기능
                    
                    **📈 재무 데이터 조회**  
                    고유명사 벡터 검색으로 정확한 회사/항목명 매칭
                    
                    **🔍 비교 분석**  
                    여러 회사의 재무 지표를 동시에 비교
                    
                    **💡 일반 상식**  
                    재무/회계 관련 기본 개념 친절하게 설명
                    
                    **🌐 최신 정보**  
                    웹 검색을 통한 실시간 시장 동향 제공
                    
                    **🧠 Adaptive RAG**  
                    질문 유형에 따라 자동으로 최적 전략 선택
                    
                    </div>
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("""
                    <div class="info-box">
                    
                    ## ✨ 기술 스택
                    
                    ✅ **벡터 기반 고유명사 검색** - 회사명/항목명 매칭 정확도 향상  
                    ✅ **LangGraph StateGraph** - Text2SQL 파이프라인 최적화  
                    ✅ **대화 컨텍스트 유지** - Short-term Memory로 자연스러운 대화  
                    ✅ **산업별 항목명 매핑** - 제조업/금융/통신 산업 특성 반영  
                    ✅ **재무비율 자동 계산** - 영업이익률, ROE, 부채비율 등
                    
                    ## 💡 예시 질문
                    
                    🏢 **단일 조회**: "삼성전자의 2025년 상반기 매출액은?"  
                    📊 **비교 분석**: "SK텔레콤과 KT의 영업이익률을 비교해줘"  
                    🔢 **재무비율**: "삼성전자와 SK하이닉스의 ROE는?"  
                    📚 **일반 상식**: "재무상태표란 무엇인가요?"  
                    🌐 **최신 정보**: "최근 AI 기술 트렌드는?"
                    
                    </div>
                    """)
            
            gr.Markdown("---")
            
            # 채팅 인터페이스 - 더 넓고 크게
            gr.Markdown("## 💬 대화 시작하기")
            chatbot = gr.Chatbot(
                label="AI 재무 분석가와의 대화",
                height=600,
                show_label=True,
                container=True,
                bubble_full_width=True,
                avatar_images=(None, "🤖"),
                show_copy_button=True
            )
            
            # 입력 인터페이스
            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="💬 재무제표 관련 질문을 입력하세요... (예: '삼성전자와 SK하이닉스의 영업이익률을 비교해줘') - Enter 키로 전송",
                    scale=5,
                    lines=1,
                    max_lines=1
                )
                send_btn = gr.Button("📤 전송", variant="primary", scale=1, size="lg")
            
            # 예시 질문 버튼들
            gr.Markdown("### 🔍 빠른 질문 예시")
            with gr.Row():
                example_btn1 = gr.Button("📚 재무상태표란?", size="sm", scale=1)
                example_btn2 = gr.Button("📊 SK텔레콤 매출액", size="sm", scale=1)
                example_btn3 = gr.Button("🤖 최근 AI 트렌드", size="sm", scale=1)
                example_btn4 = gr.Button("⚖️ 통신 3사 비교", size="sm", scale=1)
                example_btn5 = gr.Button("💰 반도체 업체 ROE", size="sm", scale=1)
            
            # 이벤트 핸들러 설정
            def submit_message(message, history):
                return self.chat_with_system(message, history)
            
            # 전송 버튼 클릭 이벤트
            send_btn.click(
                submit_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input]
            )
            
            # 엔터 키 이벤트
            msg_input.submit(
                submit_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input]
            )
            
            # 예시 질문 버튼 이벤트
            example_btn1.click(
                lambda: ("재무상태표란 무엇인가요?", []),
                outputs=[msg_input, chatbot]
            )
            
            example_btn2.click(
                lambda: ("SK텔레콤의 2025년 상반기 매출액을 알려주세요.", []),
                outputs=[msg_input, chatbot]
            )
            
            example_btn3.click(
                lambda: ("최근 AI 기술 트렌드에 대해 알려주세요.", []),
                outputs=[msg_input, chatbot]
            )
            
            example_btn4.click(
                lambda: ("SK텔레콤, 케이티, LG유플러스의 영업이익률을 비교 분석해주세요.", []),
                outputs=[msg_input, chatbot]
            )
            
            example_btn5.click(
                lambda: ("삼성전자와 SK하이닉스의 ROE를 비교해주세요.", []),
                outputs=[msg_input, chatbot]
            )
            
            # 시스템 정보
            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 📝 시스템 사양
                    - **버전**: v2.0 Pro
                    - **LLM**: GPT-4o-mini
                    - **임베딩**: text-embedding-3-large
                    - **Vector Store**: InMemory
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 🛠️ 기술 스택
                    - **Framework**: LangChain + LangGraph
                    - **Database**: SQLite
                    - **Search**: Tavily API
                    - **UI**: Gradio 4.0+
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 📊 데이터 출처
                    - **재무제표**: DART 공시 데이터
                    - **기준일**: 2025년 상반기
                    - **산업**: 제조/금융/통신
                    - **업데이트**: 2025-10-01
                    """)
            
            gr.Markdown("""
            <div style="text-align: center; padding: 20px; color: #666;">
            <p>💡 <strong>Tip:</strong> 대화 기록이 유지되므로 후속 질문도 자연스럽게 할 수 있습니다!</p>
            <p style="font-size: 0.9em;">예: "삼성전자 매출액은?" → "순이익은?" → "SK하이닉스와 비교해줘"</p>
            </div>
            """)
        
        return interface
    
    def run(self):
        """애플리케이션을 실행합니다."""
        
        # 데이터 초기화
        print("데이터를 초기화하는 중...")
        if not self.initialize_data():
            print("경고: 데이터 초기화에 실패했습니다. 빈 데이터베이스로 시작합니다.")
        
        # Gradio 인터페이스 생성 및 실행
        interface = self.create_interface()
        
        print("Gradio 서버를 시작합니다...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )


def main():
    """메인 함수"""
    try:
        app = FinancialAnalysisApp()
        app.run()
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        print("환경 변수가 올바르게 설정되었는지 확인해주세요.")


if __name__ == "__main__":
    main()

