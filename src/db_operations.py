import sqlite3
from typing import List, Dict, Any

def get_connection(db_path: str) -> sqlite3.Connection:
    """Initialize the database with necessary tables if they don't exist."""
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
        
    c.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            user_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

def add_memory_to_db(conn: sqlite3.Connection, content: str, user_id: str = None) -> int:
    """Add a new memory to the database or update an existing one."""
    c = conn.cursor()
    
    # Check if user already has a memory entry
    c.execute("SELECT 1 FROM memories WHERE user_id = ?", (user_id,))
    exists = c.fetchone()
    
    if exists:
        # Update existing memory
        c.execute(
            "UPDATE memories SET content = ?, timestamp = CURRENT_TIMESTAMP WHERE user_id = ?",
            (content, user_id)
        )
    else:
        # Insert new memory
        c.execute(
            "INSERT INTO memories (user_id, content) VALUES (?, ?)",
            (user_id, content)
        )
    
    conn.commit()
    return c.lastrowid

def get_memories_by_userid(conn: sqlite3.Connection, user_id: str) -> str:
    """Retrieve all memories for a specific user_id."""
    c = conn.cursor()
    
    try:
        c.execute(
            "SELECT content FROM memories WHERE user_id = ?",
            (user_id,)
        )
        
        # Get the single memory for this user
        row = c.fetchone()
        return row[0] if row else ""

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "No memories yet!"