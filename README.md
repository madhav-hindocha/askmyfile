---
title: AskMyFile
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AskMyFile

Upload any file. Ask a question. Get an answer.

**Two ways to run:**
- **Locally** — runs entirely on your own computer using a local Ollama
  model: no cloud AI, no API keys, no subscriptions. (Double-click `run.bat`.)
- **Hosted online** — set a `GROQ_API_KEY` environment variable and the app
  automatically uses Groq's fast cloud AI instead of Ollama, so it works on
  a web host (e.g. Hugging Face Spaces) with no local model. See
  `DEPLOY.md`.

## Getting the project from GitHub (fresh computer)

No commands needed:

1. On the GitHub page, click **Code → Download ZIP**
2. Unzip the folder anywhere
3. Double-click **run.bat**

`run.bat` sets up EVERYTHING automatically: Python and Ollama (installed
via winget if missing — if it installs one, it will ask you to
double-click run.bat once more), the Tesseract OCR program, the Python
environment with exact pinned package versions, and the AI model listed
in `.env` (qwen3:8b, ~5 GB download, one time). The first run takes a
while; every run after that starts in seconds.

(Prefer git? `git clone https://github.com/<your-username>/askmyfile.git`
works the same way.)

Accounts, uploaded files, and logs are NOT in the repo (they're
personal data, excluded by `.gitignore`) — a fresh copy starts clean
and creates them as it runs.

## One-click run

**Windows:** double-click `run.bat`
**Mac/Linux:** open a terminal in this folder and run `./run.sh` (first
time only: `chmod +x run.sh`)

**Before your first run**, install these two things once:
1. **Python** — https://www.python.org/downloads/ (tick "Add python.exe to
   PATH" during install on Windows)
2. **Ollama** — https://ollama.com (just run the installer)

## Signing in

Sign up and log in with **email + password**. No other setup needed.

## How answering works

Upload a file → ask a question → the app checks how closely your question
matches the file's content → if it's a good match, the answer is grounded
in your file; otherwise the AI answers from its own general knowledge.
Either way you always get a real answer.

You can tune how strict the "is this about my file" check is by adding
`RELEVANCE_THRESHOLD` to `.env` (default `0.35`; lower = more questions
get treated as file-related, higher = stricter).

## OCR support (scanned PDFs and images)

Scanned PDFs (no text layer) and images (JPG, PNG, etc.) are read with
OCR. The Python packages install automatically; the underlying
**Tesseract OCR** program is also needed, and `run.bat` tries to install
it automatically via winget. If that fails, install it once manually:

- Windows: https://github.com/UB-Mannheim/tesseract/wiki (installer)
- Mac: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

OCR works best on clear, high-contrast scans — a blurry photo may not
extract cleanly, but you'll get a clear message either way.

## What changed in this version

- **Scanned PDFs now work.** PDFs with no text layer are automatically
  rendered page-by-page and read with OCR instead of being rejected.
- **OCR is on by default** — `pytesseract`/`Pillow` install
  automatically, and `run.bat` auto-installs the Tesseract program.
- **"Continue with Google" is removed completely.** Sign-in is email +
  password only — no Google credentials, no `.env` OAuth setup, no
  `authlib` dependency, no setup page.
- **The "answered from general knowledge" note is removed.** Questions
  outside your file are still answered normally; there's just no tag
  under the answer anymore.

Everything else (Excel/HTML-table fallback, .ipynb support, silent
desktop-icon launcher, model warm-up and keep-alive speedups) is
unchanged from the previous version.

## Manual setup (if you don't want to use the one-click scripts)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate      # Mac/Linux

pip install -r requirements.txt
ollama pull llama3.2:1b

python app.py
```
Then open http://127.0.0.1:5000

## Folder structure

```
askmyfile/
├── run.bat                          # Manual/debug launcher, Windows (shows a console)
├── run.sh                           # One-click launcher, Mac/Linux
├── run_silent.bat                   # Same as run.bat, but silent -- used by the Desktop icon
├── launch_silent.vbs                # Runs run_silent.bat with a fully hidden window
├── create_desktop_shortcut.bat      # Creates a Windows Desktop icon
├── create_desktop_shortcut.sh       # Creates a Mac/Linux Desktop icon
├── assets/                          # App icons
├── app.py                           # Flask backend: auth + upload/ask routes
├── auth.py                          # Email + password account logic
├── rag_utils.py                     # RAG pipeline: extraction, chunking, embeddings, FAISS, Ollama
├── requirements.txt
├── .env.example                     # Copy to .env to change model/threshold
├── templates/
│   ├── login.html
│   ├── signup.html
│   └── index.html
├── static/
│   └── style.css
├── uploads/                         # Uploaded files + each user's saved FAISS index
├── logs/
│   └── startup.log                  # Written by run_silent.bat
└── users.db                         # Auto-created SQLite database of accounts
```

## Supported file types

PDF (including scanned PDFs via OCR), Word (DOCX), Excel (XLSX/XLS), CSV,
TXT, MD, JSON, Jupyter notebooks (.ipynb), images (JPG/PNG/etc. via OCR),
and most other plain-text-like files.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Python was not found" / "Ollama was not found" | Install from the links above, then run the launcher again |
| Browser says connection refused | Wait for "Server is starting!" to finish — the very first run also downloads the AI model. If it still fails, run `python app.py` manually (after activating `venv`) to see the exact error |
| `pip install` fails partway through | Delete the `venv` folder and re-run the launcher for a clean install |
| Excel upload fails | The app auto-retries other engines, but a corrupted or password-protected file will still fail |
| Scanned PDF / image upload says OCR isn't set up | Install Tesseract (see "OCR support" above), then run `run.bat` again |
| "Your session expired" | Just log in again — sessions reset when the server restarts |
| Desktop icon doesn't seem to do anything | Give it 10-15 seconds; if nothing happens, check `logs\startup.log` or run `run.bat` directly |

## Notes

This is a simple, from-scratch project meant for local/personal use or a
college project — it does not include things like password reset, email
verification, or multi-document libraries per user (one active file per
user, by design, to keep things simple and fast).
