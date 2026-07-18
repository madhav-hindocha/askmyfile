"""
auth.py
-------
Account system for AskMyFile: email + password sign-in, with a username
and an optional profile photo per account.

Accounts are stored in a local SQLite database. Passwords are hashed with
Werkzeug's password hashing (bundled with Flask) -- never stored in plain
text.
"""

import re
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "users.db"

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,20}$")


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Creates the users table on first run, and MIGRATES older databases:
    if the table already exists without the newer username/photo columns,
    they're added in place so existing accounts keep working.
    """
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            auth_provider TEXT NOT NULL DEFAULT 'local',
            display_name TEXT,
            username TEXT,
            photo TEXT
        )
    """)
    existing = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
    if "username" not in existing:
        conn.execute("ALTER TABLE users ADD COLUMN username TEXT")
    if "photo" not in existing:
        conn.execute("ALTER TABLE users ADD COLUMN photo TEXT")
    conn.commit()
    conn.close()


def _normalize_email(email):
    return email.strip().lower()


def is_valid_email(email):
    return bool(EMAIL_PATTERN.match(email.strip()))


def is_valid_username(username):
    return bool(USERNAME_PATTERN.match(username.strip()))


def check_password_strength(password):
    """
    Professional-grade minimum: 8+ characters with at least one letter
    and one number. Returns (ok: bool, message: str).
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain at least one letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."
    return True, ""


def email_exists(email):
    """Used for live 'already registered' feedback on the signup page."""
    conn = _connect()
    row = conn.execute(
        "SELECT 1 FROM users WHERE email = ?", (_normalize_email(email),)
    ).fetchone()
    conn.close()
    return row is not None


def username_exists(username):
    """Used for live 'username taken' feedback on the signup page."""
    conn = _connect()
    row = conn.execute(
        "SELECT 1 FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),)
    ).fetchone()
    conn.close()
    return row is not None


def create_user(email, username, password):
    """
    Creates a new account with email, username, and password.
    Returns (success: bool, message: str).
    """
    email = _normalize_email(email)
    username = username.strip()

    if not email or not username or not password:
        return False, "Email, username, and password are all required."
    if not is_valid_email(email):
        return False, "Please enter a valid email address."
    if not is_valid_username(username):
        return False, "Username must be 3-20 characters: letters, numbers, and underscores only."
    ok, message = check_password_strength(password)
    if not ok:
        return False, message
    if username_exists(username):
        return False, "That username is taken. Try another."

    conn = _connect()
    try:
        password_hash = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (email, password_hash, auth_provider, username) VALUES (?, ?, 'local', ?)",
            (email, password_hash, username)
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with that email already exists. Try logging in instead."
    finally:
        conn.close()


def verify_user(email, password):
    """
    Checks an email/password pair.
    Returns True if valid, False otherwise.
    """
    conn = _connect()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE email = ? AND auth_provider = 'local'",
        (_normalize_email(email),)
    ).fetchone()
    conn.close()

    if row is None or row[0] is None:
        return False
    return check_password_hash(row[0], password)


def get_user(email):
    """
    Returns {email, username, photo} for an account, or None.
    Older accounts created before usernames existed get a sensible
    fallback (the part of their email before the @).
    """
    conn = _connect()
    row = conn.execute(
        "SELECT email, username, photo FROM users WHERE email = ?",
        (_normalize_email(email),)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "email": row[0],
        "username": row[1] or row[0].split("@")[0],
        "photo": row[2],
    }


def update_username(email, username):
    """
    Changes an account's username (used on the Profile page).
    Returns (success: bool, message: str).
    """
    username = username.strip()
    if not is_valid_username(username):
        return False, "Username must be 3-20 characters: letters, numbers, and underscores only."

    conn = _connect()
    try:
        current = conn.execute(
            "SELECT LOWER(username) FROM users WHERE email = ?",
            (_normalize_email(email),)
        ).fetchone()
        taken = conn.execute(
            "SELECT 1 FROM users WHERE LOWER(username) = LOWER(?) AND email != ?",
            (username, _normalize_email(email))
        ).fetchone()
        if taken:
            return False, "That username is taken. Try another."
        conn.execute(
            "UPDATE users SET username = ? WHERE email = ?",
            (username, _normalize_email(email))
        )
        conn.commit()
        return True, "Username updated."
    finally:
        conn.close()


def set_photo(email, photo_filename):
    """Saves the filename of the user's uploaded profile photo."""
    conn = _connect()
    conn.execute(
        "UPDATE users SET photo = ? WHERE email = ?",
        (photo_filename, _normalize_email(email))
    )
    conn.commit()
    conn.close()
