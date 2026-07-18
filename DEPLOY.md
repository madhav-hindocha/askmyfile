# Deploying AskMyFile online (Hugging Face Spaces + Groq)

This makes AskMyFile run **live on the web** so anyone can open a link and
use it — no download, no install. It uses:

- **Groq** for the AI (free, fast cloud API — replaces local Ollama)
- **Hugging Face Spaces** for hosting (free tier)

The app already supports this: when a `GROQ_API_KEY` is present it uses
Groq automatically; otherwise it keeps using local Ollama. So the same
code runs both locally and online.

---

## Step 1 — Get a free Groq API key

1. Go to https://console.groq.com and sign up (free).
2. Open **API Keys** → **Create API Key**.
3. Copy the key (starts with `gsk_...`). Keep it private — treat it like a
   password. **Do NOT commit it to GitHub or put it in `.env`.**

## Step 2 — Create a Hugging Face account + Space

1. Sign up at https://huggingface.co (free).
2. Click your avatar → **New Space**.
3. Fill in:
   - **Owner**: your username
   - **Space name**: `askmyfile`
   - **License**: your choice
   - **SDK**: choose **Docker** → **Blank**
   - **Visibility**: Public
4. Click **Create Space**. You now have an empty Space with its own Git repo.

## Step 3 — Add your Groq key as a secret (never in code)

1. In the Space, go to **Settings** → **Variables and secrets**.
2. Click **New secret**:
   - **Name**: `GROQ_API_KEY`
   - **Value**: paste your `gsk_...` key
3. (Recommended) Add a second secret so logins survive restarts:
   - **Name**: `SECRET_KEY`
   - **Value**: any long random string (e.g. from a password generator)

## Step 4 — Upload the app files to the Space

The Space needs these files (all already in this folder):
`Dockerfile`, `requirements.txt`, `app.py`, `auth.py`, `rag_utils.py`,
`README.md`, `templates/`, `static/`, and the empty `uploads/` + `logs/`.

**Easiest (web upload):**
1. In the Space, click **Files** → **Add file** → **Upload files**.
2. Drag in everything from this folder EXCEPT: `venv/`, `users.db`,
   `.secret_key`, `run.bat` (not needed online).
3. Commit.

**Or via Git (if you prefer the terminal):**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/askmyfile
# copy the app files into that folder, then:
cd askmyfile
git add .
git commit -m "Add AskMyFile app"
git push
```

## Step 5 — Watch it build

The Space automatically starts **Building** (installing Tesseract, then the
Python packages — this takes several minutes the first time because of
PyTorch). When it finishes, the status turns to **Running** and your app is
live at:

**https://YOUR_USERNAME-askmyfile.hf.space**

Share that link — it opens and runs in the browser, one click.

---

## Important notes

- **Storage is temporary.** On the free tier, uploaded files, user
  accounts, and indexes are wiped whenever the Space restarts or rebuilds.
  Fine for a demo/portfolio; for permanent accounts you'd need paid
  persistent storage or an external database.
- **Free Groq limits.** The free tier has generous per-minute limits that
  are plenty for personal/demo use. Heavy traffic may hit rate limits.
- **Change the AI model** by adding a `GROQ_MODEL` secret (e.g.
  `openai/gpt-oss-120b` for higher quality, or `llama-3.1-8b-instant`).
  Default is `openai/gpt-oss-20b`.
- **Your local setup is unchanged.** Running `run.bat` locally still uses
  Ollama, because no `GROQ_API_KEY` is set on your machine.
