"""
app.py
------
Flask backend for AskMyFile.

Pipeline:
File Upload -> Text Extraction -> Chunking -> Embeddings -> FAISS Storage
-> (on question) Similarity Search -> Context Retrieval -> LLM (Groq/Ollama) -> Answer

Sign-in: email + password.

Each account has its own private uploaded file, FAISS index, and folder on
disk -- one user can never see another's data.
"""

import os
import socket

# Force IPv4 DNS resolution to avoid EAI_NODATA / [Errno -5] errors on some hosts (like Render)
_orig_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)
socket.getaddrinfo = _patched_getaddrinfo

import re
import time
from datetime import timedelta
from functools import wraps
from dotenv import load_dotenv

# Must load .env BEFORE importing rag_utils, since it reads OLLAMA_MODEL
# from the environment at import time.
load_dotenv()

from flask import (
    Flask, request, jsonify, render_template, redirect, url_for, session,
    send_from_directory, Response, stream_with_context,
)
from werkzeug.utils import secure_filename

from rag_utils import (
    extract_text, chunk_text, VectorStore, ask_llm, ask_llm_stream,
    warm_up_model, start_background_loading, RELEVANCE_THRESHOLD,
    mentions_the_file, sounds_like_no_answer_found,
)
import auth

# ---------------------------------------------------------------------------
# LIVE RELOAD: with USE_RELOADER on, saving any .py file restarts the
# server automatically -- no more closing the console and re-running
# run.bat after every code change. Flask runs TWO processes in this mode
# (a watcher + the actual server); the check below makes sure the heavy
# startup work (loading AI models) only happens in the serving process.
# ---------------------------------------------------------------------------
USE_RELOADER = True

# True when running on a host (Hugging Face Spaces etc.), which sets PORT.
# In that case there's no auto-reloader, so this single process serves.
IS_DEPLOYED = "PORT" in os.environ


def _is_serving_process():
    # When deployed there's only one process (no reloader), so it always
    # serves. Locally with the reloader on, only the child process (marked
    # by WERKZEUG_RUN_MAIN) does the heavy model loading.
    if IS_DEPLOYED or not USE_RELOADER:
        return True
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"


app = Flask(__name__)
# In debug mode, don't let the browser cache style.css/theme.js -- design
# changes show up on a normal refresh instead of needing Ctrl+Shift+R.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


def _get_secret_key():
    """
    A stable secret key so logins SURVIVE server restarts (previously a
    random key was generated on every start, which invalidated all
    sessions -- the cause of constant "session expired" messages).
    Priority: SECRET_KEY in .env > a local .secret_key file (created
    automatically on first run).
    """
    key = os.environ.get("SECRET_KEY", "").strip()
    if key:
        return key
    key_file = ".secret_key"
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            saved = f.read().strip()
        if saved:
            return saved
    key = os.urandom(32).hex()
    with open(key_file, "w", encoding="utf-8") as f:
        f.write(key)
    return key


app.secret_key = _get_secret_key()
# "Remember me" sessions last 30 days; normal sessions end when the
# browser closes.
app.permanent_session_lifetime = timedelta(days=30)

# ---------------------------------------------------------------------------
# Simple in-memory login rate limiting: after 5 wrong passwords for the
# same email, that email is locked out for 60 seconds. Protects against
# someone brute-forcing a password without annoying normal users.
# ---------------------------------------------------------------------------
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
_failed_logins = {}  # email -> {"count": int, "locked_until": float}


def _login_locked(email):
    entry = _failed_logins.get(email)
    if not entry:
        return 0
    remaining = int(entry["locked_until"] - time.time())
    return max(0, remaining)


def _record_failed_login(email):
    entry = _failed_logins.setdefault(email, {"count": 0, "locked_until": 0})
    entry["count"] += 1
    if entry["count"] >= MAX_LOGIN_ATTEMPTS:
        entry["locked_until"] = time.time() + LOCKOUT_SECONDS
        entry["count"] = 0


def _clear_failed_logins(email):
    _failed_logins.pop(email, None)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

auth.init_db()
if _is_serving_process():
    start_background_loading()
    warm_up_model()

# Each logged-in user gets their own vector store + filename, kept in memory
# for as long as the server runs. Key = normalized email.
user_sessions = {}

def _safe_folder_name(email):
    """Turns an email into a filesystem-safe folder name."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", email)


def user_folder(email):
    return os.path.join(UPLOAD_FOLDER, _safe_folder_name(email))


def get_user_session():
    """
    Returns the current user's private session data, creating it if
    needed. On first access after a server restart, tries to reload a
    previously saved FAISS index from disk so the user doesn't have to
    re-upload their file.
    """
    email = session["user_email"]
    if email not in user_sessions:
        store = VectorStore()
        folder = user_folder(email)
        loaded = store.load(folder)

        filename = None
        if loaded:
            meta_path = os.path.join(folder, "filename.txt")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    filename = f.read().strip()

        user_sessions[email] = {"vector_store": store, "filename": filename}
    return user_sessions[email]


def login_required_page(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_email" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def login_required_api(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_email" not in session:
            return jsonify({"success": False, "message": "Your session expired. Please log in again."}), 401
        return view_func(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# AUTH ROUTES -- email + password
# ---------------------------------------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_email" in session:
        return redirect(url_for("home"))
    if request.method == "GET":
        return render_template("signup.html")

    email = request.form.get("email", "")
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")

    if password != confirm:
        return render_template("signup.html", error="Passwords don't match. Please retype them.", email=email, username=username)

    success, message = auth.create_user(email, username, password)
    if success:
        session["user_email"] = email.strip().lower()
        session.permanent = True
        return redirect(url_for("home"))
    return render_template("signup.html", error=message, email=email, username=username)


@app.route("/check-email")
def check_email():
    """Live 'already registered' check used by the signup page."""
    email = request.args.get("email", "").strip()
    if not auth.is_valid_email(email):
        return jsonify({"status": None})
    return jsonify({"status": "taken" if auth.email_exists(email) else "available"})


@app.route("/check-username")
def check_username():
    """Live 'username taken' check used by the signup page."""
    username = request.args.get("username", "").strip()
    if not auth.is_valid_username(username):
        return jsonify({"status": None})
    return jsonify({"status": "taken" if auth.username_exists(username) else "available"})


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_email" in session:
        return redirect(url_for("home"))
    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    remember = request.form.get("remember") == "on"

    locked_for = _login_locked(email)
    if locked_for:
        return render_template(
            "login.html",
            error=f"Too many wrong attempts. Try again in {locked_for} seconds.",
            email=email,
        )

    if auth.verify_user(email, password):
        _clear_failed_logins(email)
        session["user_email"] = email
        session.permanent = remember
        return redirect(url_for("home"))

    _record_failed_login(email)
    return render_template("login.html", error="Incorrect email or password.", email=email)


@app.route("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# MAIN APP ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def landing():
    """Public landing/home page. Logged-in users go straight to the app."""
    if "user_email" in session:
        return redirect(url_for("home"))
    return render_template("landing.html")


@app.route("/app")
@login_required_page
def home():
    user_data = get_user_session()
    user = auth.get_user(session["user_email"])
    return render_template(
        "index.html",
        user=user,
        filename=user_data["filename"],
    )


# ---------------------------------------------------------------------------
# PROFILE -- username + photo
# ---------------------------------------------------------------------------
ALLOWED_PHOTO_EXTS = {"png", "jpg", "jpeg", "webp", "gif"}
MAX_PHOTO_BYTES = 3 * 1024 * 1024  # 3 MB


@app.route("/profile", methods=["GET", "POST"])
@login_required_page
def profile():
    email = session["user_email"]
    if request.method == "POST":
        ok, msg = auth.update_username(email, request.form.get("username", ""))
        user = auth.get_user(email)
        return render_template("profile.html", user=user, message=msg, success=ok)
    return render_template("profile.html", user=auth.get_user(email))


@app.route("/profile/photo", methods=["POST"])
@login_required_page
def upload_photo():
    email = session["user_email"]
    user = auth.get_user(email)
    file = request.files.get("photo")

    if not file or file.filename == "":
        return render_template("profile.html", user=user, message="Please choose an image first.", success=False)

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_PHOTO_EXTS:
        return render_template("profile.html", user=user, message="Please upload a PNG, JPG, WEBP, or GIF image.", success=False)

    if request.content_length and request.content_length > MAX_PHOTO_BYTES:
        return render_template("profile.html", user=user, message="That image is too large -- 3 MB max.", success=False)

    folder = user_folder(email)
    os.makedirs(folder, exist_ok=True)
    # Remove any previous profile photo (may have a different extension).
    for old in os.listdir(folder):
        if old.startswith("profile."):
            try:
                os.remove(os.path.join(folder, old))
            except OSError:
                pass

    filename = f"profile.{ext}"
    file.save(os.path.join(folder, filename))
    auth.set_photo(email, filename)

    user = auth.get_user(email)
    return render_template("profile.html", user=user, message="Profile photo updated!", success=True)


@app.route("/avatar")
@login_required_page
def avatar():
    """Serves the logged-in user's profile photo (or the default icon)."""
    user = auth.get_user(session["user_email"])
    if user and user["photo"]:
        folder = user_folder(session["user_email"])
        if os.path.exists(os.path.join(folder, user["photo"])):
            return send_from_directory(folder, user["photo"])
    return redirect(url_for("static", filename="default-avatar.svg"))


@app.route("/upload", methods=["POST"])
@login_required_api
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Please choose a file first."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"success": False, "message": "That filename isn't valid. Please rename the file and try again."}), 400

    user_data = get_user_session()
    folder = user_folder(session["user_email"])
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    file.save(file_path)

    try:
        raw_text = extract_text(file_path)
        chunks = chunk_text(raw_text)
        user_data["vector_store"].build(chunks)
        user_data["filename"] = filename

        user_data["vector_store"].save(folder)
        with open(os.path.join(folder, "filename.txt"), "w", encoding="utf-8") as f:
            f.write(filename)

        return jsonify({
            "success": True,
            "message": f"'{filename}' processed successfully. {len(chunks)} chunks indexed.",
            "filename": filename,
        })

    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error processing file: {str(e)}"}), 500


@app.route("/clear-file", methods=["POST"])
@login_required_api
def clear_file():
    user_data = get_user_session()
    user_data["vector_store"] = VectorStore()
    user_data["filename"] = None
    return jsonify({"success": True})


@app.route("/ask", methods=["POST"])
@login_required_api
def ask_question():
    data = request.get_json()
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"success": False, "message": "Please type a question."}), 400

    user_data = get_user_session()

    try:
        relevant_chunks, scores = [], []
        has_file = user_data["filename"] is not None
        if has_file:
            relevant_chunks, scores = user_data["vector_store"].search(question, top_k=5)

        best_score = max(scores) if scores else -1.0

        # A question counts as "about the file" if either:
        # - its best-matching chunk clears the similarity bar, or
        # - it clearly references the file/document itself -- broad
        #   summary or count questions ("what's in this file", "how many
        #   rows are there") often don't share vocabulary with any one
        #   chunk, so similarity search alone would wrongly call them
        #   unrelated.
        file_is_relevant = has_file and (best_score >= RELEVANCE_THRESHOLD or mentions_the_file(question))

        if file_is_relevant:
            all_chunks = user_data["vector_store"].chunks
            if len(all_chunks) <= 8:
                # Small document (a marksheet, certificate, one-pager):
                # send ALL of it. Retrieval can only lose information
                # here, and totals/counts ("how many marks") need the
                # whole document to be answered correctly.
                relevant_chunks = all_chunks
            elif mentions_the_file(question):
                # Broad questions about the file (summaries, totals,
                # counts) need wider context than a pinpoint question.
                relevant_chunks, _ = user_data["vector_store"].search(question, top_k=8)

        chunks_for_answer = relevant_chunks if file_is_relevant else None

        def generate():
            """
            Streams the answer to the browser piece by piece as the model
            generates it, so text starts appearing immediately instead of
            after the whole answer is done. The first ~200 characters are
            buffered to keep the "file answer came up empty -> retry from
            general knowledge" safety net working.
            """
            buffer = ""
            flushed = False
            try:
                for piece in ask_llm_stream(question, chunks_for_answer):
                    if not flushed:
                        buffer += piece
                        # 90 chars is enough to catch "I couldn't find..."
                        # style openings while showing text ~2x sooner.
                        if len(buffer) >= 90:
                            if chunks_for_answer and sounds_like_no_answer_found(buffer):
                                for p2 in ask_llm_stream(question):
                                    yield p2
                                return
                            yield buffer
                            flushed = True
                    else:
                        yield piece
                if not flushed:
                    if chunks_for_answer and sounds_like_no_answer_found(buffer):
                        for p2 in ask_llm_stream(question):
                            yield p2
                    else:
                        yield buffer
            except Exception as e:
                yield f"\n\nSorry — something went wrong while answering: {e}"

        return Response(stream_with_context(generate()), mimetype="text/plain")

    except Exception as e:
        return jsonify({"success": False, "message": f"Error generating answer: {str(e)}"}), 500


if __name__ == "__main__":
    # PORT/HOST come from the environment when deployed (Hugging Face Spaces
    # sets PORT=7860). Locally they default to the original 127.0.0.1:5000.
    # When a PORT is provided by a host, we also turn OFF debug + the
    # auto-reloader, since those are development-only conveniences.
    port = int(os.environ.get("PORT", "5000"))
    host = "0.0.0.0" if IS_DEPLOYED else "127.0.0.1"
    app.run(
        host=host,
        port=port,
        debug=not IS_DEPLOYED,
        use_reloader=USE_RELOADER and not IS_DEPLOYED,
    )
