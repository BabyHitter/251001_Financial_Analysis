import os
import glob
from typing import List, Tuple
from database import db

class FinancialDataParser:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def parse_tsv_file(self, file_path: str) -> List[Tuple]:
        """TSV 파일을 파싱하여 데이터 튜플 리스트를 반환합니다."""
        data = []
        
        # 여러 인코딩을 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-16']
        file_content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    file_content = file.readlines()
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"파일 {file_path} 읽기 중 오류 발생: {e}")
                return data
        
        if file_content is None:
            print(f"파일 {file_path} 파싱 중 오류 발생: 지원되는 인코딩으로 읽을 수 없습니다")
            return data
        
        try:
            for line in file_content:
                line = line.strip()
                if not line:
                    continue
                
                # 탭으로 구분된 값들을 분리
                values = line.split('\t')
                
                # 빈 값들을 None으로 변환
                processed_values = []
                for value in values:
                    if value.strip() == '' or value.strip() == '-':
                        processed_values.append(None)
                    else:
                        processed_values.append(value.strip())
                
                data.append(tuple(processed_values))
                    
        except Exception as e:
            print(f"파일 {file_path} 파싱 중 오류 발생: {e}")
            
        return data
    
    def parse_balance_sheets(self):
        """재무상태표 디렉토리의 모든 파일을 파싱하여 데이터베이스에 저장합니다."""
        balance_sheets_dir = os.path.join(self.data_dir, "balance_sheets")
        
        if not os.path.exists(balance_sheets_dir):
            print(f"재무상태표 디렉토리가 존재하지 않습니다: {balance_sheets_dir}")
            return
        
        db.clear_table("balance_sheet")
        
        txt_files = glob.glob(os.path.join(balance_sheets_dir, "*.txt"))
        
        if not txt_files:
            print("재무상태표 파일이 없습니다.")
            return
        
        all_data = []
        for file_path in txt_files:
            print(f"재무상태표 파일 파싱 중: {os.path.basename(file_path)}")
            data = self.parse_tsv_file(file_path)
            # 컬럼 개수 맞추기 (15개로 고정)
            normalized_data = []
            for row in data:
                row_list = list(row)
                while len(row_list) < 15:
                    row_list.append(None)
                normalized_data.append(tuple(row_list[:15]))
            all_data.extend(normalized_data)
        
        if all_data:
            db.insert_balance_sheet_data(all_data)
    
    def parse_income_statements(self):
        """손익계산서 디렉토리의 모든 파일을 파싱하여 데이터베이스에 저장합니다."""
        income_statements_dir = os.path.join(self.data_dir, "income_statements")
        
        if not os.path.exists(income_statements_dir):
            print(f"손익계산서 디렉토리가 존재하지 않습니다: {income_statements_dir}")
            return
        
        db.clear_table("income_statement")
        
        txt_files = glob.glob(os.path.join(income_statements_dir, "*.txt"))
        
        if not txt_files:
            print("손익계산서 파일이 없습니다.")
            return
        
        all_data = []
        for file_path in txt_files:
            print(f"손익계산서 파일 파싱 중: {os.path.basename(file_path)}")
            data = self.parse_tsv_file(file_path)
            # 컬럼 개수 맞추기 (18개로 고정)
            normalized_data = []
            for row in data:
                row_list = list(row)
                while len(row_list) < 18:
                    row_list.append(None)
                normalized_data.append(tuple(row_list[:18]))
            all_data.extend(normalized_data)
        
        if all_data:
            db.insert_income_statement_data(all_data)
    
    def parse_cash_flow_statements(self):
        """현금흐름표 디렉토리의 모든 파일을 파싱하여 데이터베이스에 저장합니다."""
        cash_flow_dir = os.path.join(self.data_dir, "cash_flow_statements")
        
        if not os.path.exists(cash_flow_dir):
            print(f"현금흐름표 디렉토리가 존재하지 않습니다: {cash_flow_dir}")
            return
        
        db.clear_table("cash_flow_statement")
        
        txt_files = glob.glob(os.path.join(cash_flow_dir, "*.txt"))
        
        if not txt_files:
            print("현금흐름표 파일이 없습니다.")
            return
        
        all_data = []
        for file_path in txt_files:
            print(f"현금흐름표 파일 파싱 중: {os.path.basename(file_path)}")
            data = self.parse_tsv_file(file_path)
            # 컬럼 개수 맞추기 (16개로 고정)
            normalized_data = []
            for row in data:
                row_list = list(row)
                while len(row_list) < 16:
                    row_list.append(None)
                normalized_data.append(tuple(row_list[:16]))
            all_data.extend(normalized_data)
        
        if all_data:
            db.insert_cash_flow_data(all_data)
    
    def parse_equity_statements(self):
        """자본변동표 디렉토리의 모든 파일을 파싱하여 데이터베이스에 저장합니다."""
        equity_dir = os.path.join(self.data_dir, "equity_statements")
        
        if not os.path.exists(equity_dir):
            print(f"자본변동표 디렉토리가 존재하지 않습니다: {equity_dir}")
            return
        
        db.clear_table("statement_of_changes_in_equity")
        
        txt_files = glob.glob(os.path.join(equity_dir, "*.txt"))
        
        if not txt_files:
            print("자본변동표 파일이 없습니다.")
            return
        
        all_data = []
        for file_path in txt_files:
            print(f"자본변동표 파일 파싱 중: {os.path.basename(file_path)}")
            data = self.parse_tsv_file(file_path)
            # 컬럼 개수 맞추기 (15개로 고정)
            normalized_data = []
            for row in data:
                row_list = list(row)
                while len(row_list) < 15:
                    row_list.append(None)
                normalized_data.append(tuple(row_list[:15]))
            all_data.extend(normalized_data)
        
        if all_data:
            db.insert_equity_data(all_data)
    
    def parse_all_financial_statements(self):
        """모든 재무제표를 파싱하여 데이터베이스에 저장합니다."""
        print("=== 재무제표 데이터 파싱 시작 ===")
        
        self.parse_balance_sheets()
        self.parse_income_statements()
        self.parse_cash_flow_statements()
        # 자본변동표는 복잡한 구조로 인해 선택적으로 파싱
        # self.parse_equity_statements()
        
        print("=== 재무제표 데이터 파싱 완료 ===")

def main():
    """파서 테스트용 메인 함수"""
    parser = FinancialDataParser()
    parser.parse_all_financial_statements()

if __name__ == "__main__":
    main()

