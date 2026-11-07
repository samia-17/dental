"""
DentaPlan backend — lecture transcription agent
Run:
  pip install fastapi uvicorn python-multipart google-genai
  (also install ffmpeg on your machine)
  export GEMINI_API_KEY=your_key
  uvicorn main:app --reload
"""
import os, json, uuid, time, sqlite3, subprocess, glob, shutil
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SECONDS = 600  # 10-minute pieces keep long lectures accurate
DATA = os.getenv("DATA_DIR", "data")
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "300"))
import threading
SEM = threading.Semaphore(int(os.getenv("MAX_JOBS", "2")))  # lectures processed at the same time
os.makedirs(DATA, exist_ok=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
app = FastAPI(title="DentaPlan Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- tiny SQLite store ----------
def db():
    c = sqlite3.connect(f"{DATA}/lectures.db")
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS lectures(
        id TEXT PRIMARY KEY, title TEXT, subject TEXT, filename TEXT,
        status TEXT, progress TEXT, transcript TEXT, structured TEXT, study TEXT, error TEXT)""")
    try: c.execute("ALTER TABLE lectures ADD COLUMN student TEXT")
    except sqlite3.OperationalError: pass
    return c

def norm(s): return (s or "").strip().lower()

def update(lid, **kw):
    c = db()
    c.execute(f"UPDATE lectures SET {', '.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), lid))
    c.commit(); c.close()

# ---------- prompts ----------
TRANSCRIBE = """Transcribe this part of a dental-school lecture word for word.
Rules:
- Do NOT summarize, shorten, or skip anything, including side comments and examples.
- Keep medical and dental terms in English exactly as spoken (e.g. pulpotomy, abutment, pontic, retainer, occlusion).
- If the doctor speaks Arabic, keep the Arabic and keep English terms inside it.
- Mark unclear words as [unclear]. Output plain text only."""

STRUCTURE = """Organize this lecture transcript into clean study notes in Markdown.
Rules:
- Keep EVERY piece of information. Only add headings, paragraphs, bullet lists, and bold for key terms.
- Do not summarize and do not remove examples.
- Add a short list of 'Requirements mentioned' at the end (assignments, cases, tools, deadlines, exam hints) if the doctor mentioned any.
Transcript:
"""

STUDY = """From these lecture notes create study material. Return ONLY JSON:
{"summary": ["..."], "key_terms": [{"term":"","meaning":""}],
 "mcqs": [{"q":"","options":["A","B","C","D"],"answer":0,"explain":""}],
 "flashcards": [{"front":"","back":""}],
 "requirements": [{"text":"","due":""}]}
Use only facts that appear in the notes. About 10 MCQs and 15 flashcards.
Notes:
"""

# ---------- helpers ----------
def gen(contents, json_mode=False):
    cfg = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    for attempt in range(3):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=cfg).text
        except Exception:
            if attempt == 2: raise
            time.sleep(3 * (attempt + 1))

def upload(path):
    f = client.files.upload(file=path)
    while f.state.name == "PROCESSING":
        time.sleep(2); f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE": raise RuntimeError("Gemini could not process the file")
    return f

def split_media(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
                    "-f", "segment", "-segment_time", str(CHUNK_SECONDS), f"{outdir}/part_%03d.mp3"],
                   check=True, capture_output=True)
    return sorted(glob.glob(f"{outdir}/part_*.mp3"))

# ---------- the agent pipeline ----------
def run_pipeline(lid, path):
    with SEM:
        _run_pipeline(lid, path)

def _run_pipeline(lid, path):
    try:
        ext = path.lower().rsplit(".", 1)[-1]
        texts = []
        if ext in ("pdf", "pptx", "ppt", "txt", "md"):
            update(lid, status="working", progress="Reading document")
            texts.append(gen([upload(path), "Extract ALL the text from this document in order. Keep English terms. Do not summarize."]))
        else:
            update(lid, status="working", progress="Splitting audio")
            parts = split_media(path, f"{DATA}/{lid}_parts")
            for i, p in enumerate(parts, 1):
                update(lid, progress=f"Transcribing part {i}/{len(parts)}")
                texts.append(gen([upload(p), TRANSCRIBE]))
        transcript = "\n\n".join(texts)
        update(lid, transcript=transcript, progress="Organizing notes")

        # long transcripts are structured in blocks so nothing gets dropped
        blocks = [transcript[i:i + 12000] for i in range(0, len(transcript), 12000)]
        notes = "\n\n".join(gen(STRUCTURE + b) for b in blocks)
        update(lid, structured=notes, progress="Creating quiz and flashcards")

        study = gen(STUDY + notes[:60000], json_mode=True)
        json.loads(study)  # validate
        update(lid, study=study, status="done", progress="Done")
    except Exception as e:
        update(lid, status="error", error=str(e), progress="Failed")
    finally:
        shutil.rmtree(f"{DATA}/{lid}_parts", ignore_errors=True)
        try: os.remove(path)  # free disk space after processing
        except OSError: pass

# ---------- API ----------
@app.post("/lectures")
async def create_lecture(bg: BackgroundTasks, file: UploadFile = File(...),
                         title: str = Form(...), subject: str = Form(...), student: str = Form(...)):
    if not norm(student): raise HTTPException(400, "Student name is required")
    lid = uuid.uuid4().hex[:10]
    path = f"{DATA}/{lid}_{os.path.basename(file.filename)}"
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    if os.path.getsize(path) > MAX_MB * 1024 * 1024:
        os.remove(path)
        raise HTTPException(413, f"File too large (max {MAX_MB} MB)")
    c = db()
    c.execute("INSERT INTO lectures(id,title,subject,filename,status,progress,student) VALUES(?,?,?,?,?,?,?)",
              (lid, title, subject, file.filename, "queued", "Queued", norm(student)))
    c.commit(); c.close()
    bg.add_task(run_pipeline, lid, path)
    return {"id": lid, "status": "queued"}

@app.get("/lectures")
def list_lectures(student: str = "", subject: str | None = None):
    if not norm(student): return []
    c = db()
    q = "SELECT id,title,subject,filename,status,progress FROM lectures WHERE student=?"
    args = [norm(student)]
    if subject: q += " AND subject=?"; args.append(subject)
    rows = c.execute(q, args).fetchall()
    c.close()
    return [dict(r) for r in rows]

def owned(c, lid, student):
    r = c.execute("SELECT * FROM lectures WHERE id=?", (lid,)).fetchone()
    if not r or r["student"] != norm(student): raise HTTPException(404, "Lecture not found")
    return r

@app.get("/lectures/{lid}")
def get_lecture(lid: str, student: str = ""):
    c = db(); d = dict(owned(c, lid, student)); c.close()
    d["study"] = json.loads(d["study"]) if d["study"] else None
    return d

@app.delete("/lectures/{lid}")
def delete_lecture(lid: str, student: str = ""):
    c = db(); owned(c, lid, student)
    c.execute("DELETE FROM lectures WHERE id=?", (lid,)); c.commit(); c.close()
    return {"deleted": lid}

@app.get("/")
def index():
    return FileResponse("index.html")
