import sqlite3
import json
from datetime import datetime
import uuid
import hashlib

class CareerCounselingDB:
    def __init__(self, db_path="career_counseling.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                phone_number TEXT,
                educational_background TEXT,
                interests TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chat sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER,
                session_name TEXT DEFAULT 'Career Counseling Session',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id INTEGER,
                message_type TEXT,
                message_text TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password, full_name="", phone_number=""):
        password_hash = self.hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, phone_number)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, full_name, phone_number))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            # Create default session for the user
            self.create_session(user_id, "Career Counseling Session")
            
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    def verify_user(self, username, password):
        password_hash = self.hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, full_name, phone_number, educational_background, interests 
            FROM users 
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'phone_number': user[4],
                'educational_background': user[5],
                'interests': json.loads(user[6]) if user[6] else []
            }
        return None
    
    def get_user_by_id(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, full_name, phone_number, educational_background, interests 
            FROM users 
            WHERE id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'phone_number': user[4],
                'educational_background': user[5],
                'interests': json.loads(user[6]) if user[6] else []
            }
        return None
    
    def create_session(self, user_id, session_name="Career Counseling Session"):
        session_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_sessions (session_id, user_id, session_name)
            VALUES (?, ?, ?)
        ''', (session_id, user_id, session_name))
        
        conn.commit()
        conn.close()
        return session_id
    
    def save_message(self, session_id, user_id, message_type, message_text):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_messages (session_id, user_id, message_type, message_text)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_id, message_type, message_text))
        
        # Update session updated_at
        cursor.execute('''
            UPDATE chat_sessions 
            SET updated_at = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_user_sessions(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, session_name, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
        ''', (user_id,))
        
        sessions = cursor.fetchall()
        conn.close()
        
        return [
            {
                'session_id': session[0],
                'session_name': session[1],
                'created_at': session[2],
                'updated_at': session[3]
            }
            for session in sessions
        ]
    
    def get_chat_history(self, session_id, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT message_type, message_text, timestamp
            FROM chat_messages
            WHERE session_id = ? AND user_id = ?
            ORDER BY timestamp ASC
        ''', (session_id, user_id))
        
        messages = cursor.fetchall()
        conn.close()
        
        return [
            {
                'type': msg[0],
                'text': msg[1],
                'timestamp': msg[2]
            }
            for msg in messages
        ]
    
    def update_user_profile(self, user_id, educational_background, interests):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET educational_background = ?, interests = ?
            WHERE id = ?
        ''', (educational_background, json.dumps(interests), user_id))
        
        conn.commit()
        conn.close()
