# Text Summariser

A neural-network-powered text summarisation app with a FastAPI backend and Streamlit frontend.

## Project Structure

```
neural-network/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI application
│   │   ├── config.py       # Settings and constants
│   │   └── test_main.py    # Unit tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py              # Streamlit UI
│   └── requirements.txt
├── sample_texts/            # 3 sample texts for quick testing
├── .env.example
└── README.md
```

## Setup

### 1. Clone and configure

```bash
cp .env.example .env   # edit if needed
```

### 2. Backend

**Windows (PowerShell):**
```powershell
cd backend
if (Test-Path venv) { Remove-Item -Recurse -Force venv }
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Mac / Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The first run will download the model (~1.2 GB). After that it is cached locally.

The API will be available at `http://localhost:8000`. Test it:

```bash
curl -X POST http://localhost:8000/summarise \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text here...", "max_words": 80}'
```

### 3. Frontend

Open a **second terminal**:

**Windows (PowerShell):**
```powershell
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Mac / Linux:**
```bash
cd frontend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit UI opens at `http://localhost:8501`.

### 4. Run tests

From the **backend/** directory with the venv activated:

```bash
pytest app/test_main.py -v
```

### 5. Docker (backend only)

```bash
cd backend
docker build -t summariser-backend .
docker run -p 8000:8000 summariser-backend
```

## API Reference

### POST /summarise

**Request:**
```json
{
  "text": "Your text to summarise...",
  "max_words": 80
}
```

- `text` (required): non-empty string, truncated at 10 000 characters
- `max_words` (optional): clamped to 30-200, default 80

**Response:**
```json
{
  "summary": "The generated summary...",
  "model": "sshleifer/distilbart-cnn-12-6",
  "took_ms": 1234.5
}
```

### GET /health

Returns `{"status": "ok", "model_loaded": true}`.

## Troubleshooting

### "No matching distribution found for torch"

PyTorch publishes separate wheel files for each combination of OS, architecture,
and Python version. If `pip install` fails to find a torch wheel:

1. **Upgrade pip** — older pip versions cannot resolve newer wheel metadata:
   ```bash
   python -m pip install --upgrade pip
   ```
2. **Check your Python version** — Python 3.12 has the broadest torch wheel
   coverage. Python 3.13 is supported from torch 2.6.0 onwards. You can check
   with `python --version`.
3. **Create a fresh venv with Python 3.12** if your default Python is too new
   or too old:
   ```bash
   py -3.12 -m venv venv   # Windows (py launcher)
   python3.12 -m venv venv  # Mac / Linux
   ```
4. **CPU-only install** — if you don't have an NVIDIA GPU you can skip CUDA
   wheels entirely, which are smaller and resolve faster:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

### sentencepiece build errors on Windows

The `sentencepiece` Python package does not ship pre-built wheels for every
Python version. On Python 3.13+ it will try to compile from C++ source, which
fails on Windows unless you have Visual Studio Build Tools installed
(`WinError 2` / `FileNotFoundError` during pip install).

This project uses `sshleifer/distilbart-cnn-12-6` by default, which is a
BART-based model with a byte-level BPE tokenizer — **it does not need
sentencepiece**. The dependency has been removed from `requirements.txt`.

If you switch `MODEL_NAME` to a T5 or mBART model that requires sentencepiece:

1. **Use Python 3.12** where sentencepiece publishes pre-built Windows wheels:
   ```powershell
   py -3.12 -m venv venv
   venv\Scripts\activate
   pip install sentencepiece
   ```
2. Or stay on Python 3.13 and install
   [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the "Desktop development with C++" workload so that pip can compile
   sentencepiece from source.
