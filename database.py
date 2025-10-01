import sqlite3
import os
from typing import Optional

class FinancialDatabase:
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """데이터베이스 연결을 반환합니다."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """데이터베이스와 테이블들을 초기화합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 재무상태표 (Balance Sheet)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS balance_sheet (
                재무제표종류 TEXT,
                종목코드 TEXT,
                회사명 TEXT,
                시장구분 TEXT,
                업종 TEXT,
                업종명 TEXT,
                결산월 TEXT,
                결산기준일 TEXT,
                보고서종류 TEXT,
                통화 TEXT,
                항목코드 TEXT,
                항목명 TEXT,
                당기_반기말 REAL,
                전기말 REAL,
                전전기말 REAL,
                PRIMARY KEY (회사명, 결산기준일, 항목명)
            )
        """)
        
        # 손익계산서 (Income Statement)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS income_statement (
                재무제표종류 TEXT,
                종목코드 TEXT,
                회사명 TEXT,
                시장구분 TEXT,
                업종 TEXT,
                업종명 TEXT,
                결산월 TEXT,
                결산기준일 TEXT,
                보고서종류 TEXT,
                통화 TEXT,
                항목코드 TEXT,
                항목명 TEXT,
                당기_반기_3개월 REAL,
                당기_반기_누적 REAL,
                전기_반기_3개월 REAL,
                전기_반기_누적 REAL,
                전기 REAL,
                전전기 REAL,
                PRIMARY KEY (회사명, 결산기준일, 항목명)
            )
        """)
        
        # 현금흐름표 (Cash Flow Statement)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cash_flow_statement (
                재무제표종류 TEXT,
                종목코드 TEXT,
                회사명 TEXT,
                시장구분 TEXT,
                업종 TEXT,
                업종명 TEXT,
                결산월 TEXT,
                결산기준일 TEXT,
                보고서종류 TEXT,
                통화 TEXT,
                항목코드 TEXT,
                항목명 TEXT,
                당기_반기말 REAL,
                전기_반기말 REAL,
                전기 REAL,
                전전기 REAL,
                PRIMARY KEY (회사명, 결산기준일, 항목명)
            )
        """)
        
        # 자본변동표 (Statement of Changes in Equity)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statement_of_changes_in_equity (
                재무제표종류 TEXT,
                종목코드 TEXT,
                회사명 TEXT,
                시장구분 TEXT,
                업종 TEXT,
                업종명 TEXT,
                결산월 TEXT,
                결산기준일 TEXT,
                보고서종류 TEXT,
                통화 TEXT,
                항목코드 TEXT,
                항목명 TEXT,
                당기 REAL,
                전기 REAL,
                전전기 REAL,
                PRIMARY KEY (회사명, 결산기준일, 항목명)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"데이터베이스가 {self.db_path}에 초기화되었습니다.")
    
    def clear_table(self, table_name: str):
        """특정 테이블의 모든 데이터를 삭제합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        conn.commit()
        conn.close()
        print(f"{table_name} 테이블의 데이터가 삭제되었습니다.")
    
    def insert_balance_sheet_data(self, data: list):
        """재무상태표 데이터를 삽입합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO balance_sheet 
            (재무제표종류, 종목코드, 회사명, 시장구분, 업종, 업종명, 결산월, 결산기준일, 
             보고서종류, 통화, 항목코드, 항목명, 당기_반기말, 전기말, 전전기말)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        conn.close()
        print(f"{len(data)}개의 재무상태표 데이터가 삽입되었습니다.")
    
    def insert_income_statement_data(self, data: list):
        """손익계산서 데이터를 삽입합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO income_statement 
            (재무제표종류, 종목코드, 회사명, 시장구분, 업종, 업종명, 결산월, 결산기준일, 
             보고서종류, 통화, 항목코드, 항목명, 당기_반기_3개월, 당기_반기_누적, 
             전기_반기_3개월, 전기_반기_누적, 전기, 전전기)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        conn.close()
        print(f"{len(data)}개의 손익계산서 데이터가 삽입되었습니다.")
    
    def insert_cash_flow_data(self, data: list):
        """현금흐름표 데이터를 삽입합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO cash_flow_statement 
            (재무제표종류, 종목코드, 회사명, 시장구분, 업종, 업종명, 결산월, 결산기준일, 
             보고서종류, 통화, 항목코드, 항목명, 당기_반기말, 전기_반기말, 전기, 전전기)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        conn.close()
        print(f"{len(data)}개의 현금흐름표 데이터가 삽입되었습니다.")
    
    def insert_equity_data(self, data: list):
        """자본변동표 데이터를 삽입합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO statement_of_changes_in_equity 
            (재무제표종류, 종목코드, 회사명, 시장구분, 업종, 업종명, 결산월, 결산기준일, 
             보고서종류, 통화, 항목코드, 항목명, 당기, 전기, 전전기)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        conn.close()
        print(f"{len(data)}개의 자본변동표 데이터가 삽입되었습니다.")
    
    def get_table_info(self, table_name: str) -> list:
        """테이블의 스키마 정보를 반환합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        result = cursor.fetchall()
        conn.close()
        return result
    
    def get_all_companies(self) -> list:
        """모든 회사명 목록을 반환합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT 회사명 
            FROM income_statement 
            WHERE 회사명 IS NOT NULL
            ORDER BY 회사명
        """)
        companies = [row[0] for row in cursor.fetchall()]
        conn.close()
        return companies
    
    def get_all_items(self) -> list:
        """모든 재무항목명 목록을 반환합니다."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        all_items = set()
        
        # 손익계산서 항목
        cursor.execute("""
            SELECT DISTINCT 항목명 
            FROM income_statement 
            WHERE 항목명 IS NOT NULL
        """)
        all_items.update([row[0] for row in cursor.fetchall()])
        
        # 재무상태표 항목
        cursor.execute("""
            SELECT DISTINCT 항목명 
            FROM balance_sheet 
            WHERE 항목명 IS NOT NULL
        """)
        all_items.update([row[0] for row in cursor.fetchall()])
        
        # 현금흐름표 항목
        cursor.execute("""
            SELECT DISTINCT 항목명 
            FROM cash_flow_statement 
            WHERE 항목명 IS NOT NULL
        """)
        all_items.update([row[0] for row in cursor.fetchall()])
        
        conn.close()
        return sorted(list(all_items))

# 전역 데이터베이스 인스턴스
db = FinancialDatabase()

