import sqlite3
import json
from datetime import datetime

DB_PATH = "research_assistant.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Chat history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT,
            mode TEXT,
            messages TEXT,
            created_at TEXT
        )
    ''')

    # Reports table
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            report TEXT,
            web_findings TEXT,
            doc_findings TEXT,
            created_at TEXT
        )
    ''')

    conn.commit()
    conn.close()

def save_chat(session_name, mode, messages):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_sessions (session_name, mode, messages, created_at)
        VALUES (?, ?, ?, ?)
    ''', (session_name, mode, json.dumps(messages), datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_chats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, session_name, mode, created_at FROM chat_sessions ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return rows

def load_chat_by_id(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT messages FROM chat_sessions WHERE id = ?', (chat_id,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else []

def save_report(topic, report, web_findings, doc_findings):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO reports (topic, report, web_findings, doc_findings, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (topic, report, web_findings, doc_findings, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_reports():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, topic, created_at FROM reports ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return rows

def load_report_by_id(report_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    row = c.fetchone()
    conn.close()
    return row

def delete_chat(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM chat_sessions WHERE id = ?', (chat_id,))
    conn.commit()
    conn.close()

def delete_report(report_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM reports WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()