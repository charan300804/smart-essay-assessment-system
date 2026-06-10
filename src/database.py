import sqlite3
import os
from datetime import datetime

DB_NAME = "seas.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            score REAL,
            timestamp TEXT,
            word_count INTEGER,
            badges TEXT,
            snippet TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_essay(score, word_count, badges, snippet):
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    # Store badges as a comma-separated string
    badges_str = ",".join([b[0] for b in badges]) if badges else ""
    
    c.execute('INSERT INTO history (score, timestamp, word_count, badges, snippet) VALUES (?, ?, ?, ?, ?)',
              (score, timestamp, word_count, badges_str, snippet))
    conn.commit()
    conn.close()

def get_history():
    conn = get_db_connection()
    history = conn.execute('SELECT * FROM history ORDER BY id DESC').fetchall()
    conn.close()
    
    # Convert rows to list of dicts for compatibility with app.py
    return [
        {
            "score": row['score'],
            "time": row['timestamp'],
            "words": row['word_count'],
            "badges": [(b, "badge-blue") for b in row['badges'].split(",")] if row['badges'] else [], # Reconstruct badges roughly
            "snippet": row['snippet']
        }
        for row in history
    ]

def clear_history():
    conn = get_db_connection()
    conn.execute('DELETE FROM history')
    conn.commit()
    conn.close()
