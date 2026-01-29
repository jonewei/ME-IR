import sqlite3
import logging
from pathlib import Path

class FormulaIndexer:
    def __init__(self, db_path="artifacts/formula_index.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 创建索引表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS formula_index (
                formula_id TEXT PRIMARY KEY,
                h_latex TEXT,
                h_dna TEXT
            )
        ''')
        # 创建高性能检索索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_latex ON formula_index(h_latex)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dna ON formula_index(h_dna)')
        conn.commit()
        conn.close()

    def save_batch(self, batch_data):
        """批量插入数据 (formula_id, h_latex, h_dna)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.executemany('INSERT OR REPLACE INTO formula_index VALUES (?, ?, ?)', batch_data)
            conn.commit()
        finally:
            conn.close()

    def retrieve(self, q_h_latex, q_h_dna):
        """双路召回"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = 'SELECT formula_id FROM formula_index WHERE h_latex = ? OR h_dna = ?'
        cursor.execute(query, (q_h_latex, q_h_dna))
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return list(set(results))