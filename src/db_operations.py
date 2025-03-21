import os
import sqlite3
from typing import Optional

from pveagle import EagleProfile


def get_connection(db_path: str) -> sqlite3.Connection:
    """Initialize the database with necessary tables if they don't exist."""
    
    # Create the directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
        
    c.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            user_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
            CREATE TABLE IF NOT EXISTS eagle_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data BLOB NOT NULL
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

def get_memories_by_userid(conn: sqlite3.Connection, user_id: str) -> Optional[str]:
    """Retrieve all memories for a specific user_id."""
    c = conn.cursor()
    
    try:
        c.execute(
            "SELECT content FROM memories WHERE user_id = ?",
            (user_id,)
        )
        
        # Get the single memory for this user
        row = c.fetchone()
        return row[0] if row else None

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "Could not retrieve memories"

def insert_eagle_profile(conn: sqlite3.Connection, user_id: str, profile: EagleProfile) -> None:
    """Insert a eagle profile for a given user."""
    c = conn.cursor()
    profile_bytes = profile.to_bytes()

    try:

        c.execute("""
            INSERT OR REPLACE INTO eagle_profiles (user_id, profile_data)
            VALUES (?, ?)
        """, (user_id, profile_bytes))
        conn.commit()

    except sqlite3.Error as e:
        print(f"Database error: {e}")

def fetch_all_profiles(conn: sqlite3.Connection) -> dict[str, EagleProfile]:
    """Retrieve all eagle profiles of all users."""
    c = conn.cursor()

    try:

        c.execute('SELECT user_id, profile_data FROM eagle_profiles')
        rows = c.fetchall()
        profiles = {}
        for user_id, profile_data in rows:
            profile = EagleProfile.from_bytes(profile_data)
            profiles[user_id] = profile
        return profiles

    except sqlite3.Error as e:
        print(f"Database error: {e}")