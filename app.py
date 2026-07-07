from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import learner model functionality
from learner_model import (
    LearnerModel, LearnerModelManager, QuizResult, FlashcardResult, Recommendation
)

# Import comparison engine functionality
from compare_engine import (
    CompareRequest, CompareResult, ComparisonEngine
)

# Import curriculum builder functionality
from curriculum_builder import (
    BuildCurriculumRequest, Curriculum, CurriculumBuilder
)

# Import document ingestion functionality
from document_ingestion import create_document_ingester, ExtractResult

# Import recommendation functionality
from recommendation_models import (
    RecommendationFilters,
    RecommendedItem,
    RecommendationResult,
    ImportRequest,
    ImportResult,
)
from recommendation_engine import RecommendationEngine
from recommendation_import import create_import_handler, ImportHandler

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
WORKSPACE_DIR = BASE_DIR / "workspace_data"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
TRANSCRIPTS_DIR = WORKSPACE_DIR / "transcripts"
EMBEDDINGS_DIR = WORKSPACE_DIR / "embeddings"
SOURCES_FILE = WORKSPACE_DIR / "sources.json"

for d in (WORKSPACE_DIR, UPLOADS_DIR, TRANSCRIPTS_DIR, EMBEDDINGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _read_sources() -> list[dict[str, Any]]:
    if not SOURCES_FILE.exists():
        return []
    try:
        return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_sources(sources: list[dict[str, Any]]) -> None:
    SOURCES_FILE.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_context_for_llm(citations: list[dict[str, Any]], limit_chars: int = 12000) -> str:
    chunks: list[str] = []
    for c in citations:
        name = str(c.get("name") or "")
        content = str(c.get("content") or "")
        content = content.strip()
        if not content:
            continue
        header = f"Document: {name}" if name else "Document"
        chunks.append(f"{header}\n{content}")
    out = "\n\n---\n\n".join(chunks).strip()
    return out[: max(0, int(limit_chars))]


def _safe_json_loads(text: str) -> Any:
    t = (text or "").strip()
    if not t:
        return None
    # Try to extract a JSON object/array if model added prose.
    first_brace = min([i for i in [t.find("{"), t.find("[")] if i != -1], default=-1)
    if first_brace > 0:
        t = t[first_brace:]
    last_brace = max(t.rfind("}"), t.rfind("]"))
    if last_brace != -1:
        t = t[: last_brace + 1]
    try:
        return json.loads(t)
    except Exception:
        return None


def _extract_youtube_id(url: str) -> str | None:
    u = (url or "").strip()
    if not u:
        return None
    # Common forms:
    # - https://www.youtube.com/watch?v=VIDEOID
    # - https://youtu.be/VIDEOID
    # - https://www.youtube.com/shorts/VIDEOID
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", u)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", u)
    if m:
        return m.group(1)
    m = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", u)
    if m:
        return m.group(1)
    return None


def _update_source_fields(source_id: str, updates: dict[str, Any]) -> None:
    sources = _read_sources()
    changed = False
    for i, s in enumerate(sources):
        if str(s.get("id")) == str(source_id):
            sources[i] = {**s, **updates}
            changed = True
            break
    if changed:
        _write_sources(sources)


def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return
    raise RuntimeError(
        "ffmpeg is required for audio transcription fallback. Install FFmpeg and ensure 'ffmpeg' is on PATH."
    )


def _trim_audio_for_transcription(audio_path: Path, max_seconds: int) -> Path:
    if max_seconds <= 0:
        return audio_path
    trimmed = audio_path.with_name(audio_path.stem + f"_trim{max_seconds}" + audio_path.suffix)
    # Only create if not already present.
    if trimmed.exists():
        return trimmed
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return audio_path
    # -y overwrite, -t limits duration.
    import subprocess

    subprocess.run(
        [ffmpeg, "-y", "-i", str(audio_path), "-t", str(int(max_seconds)), "-ac", "1", "-ar", "16000", str(trimmed)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return trimmed if trimmed.exists() else audio_path


def _is_rate_limited_error(e: Exception) -> bool:
    msg = str(e)
    return " 429" in msg or "HTTP Error 429" in msg or "Too Many Requests" in msg


def _download_youtube_audio(url: str, out_dir: Path, source_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / f"{source_id}.%(ext)s")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        # Extract to mp3 for easier decoding downstream. Requires ffmpeg.
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    from yt_dlp import YoutubeDL

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            last_err = None
            break
        except Exception as e:
            last_err = e
            # Backoff on rate limits
            time.sleep(2.0 + attempt * 3.0)
    if last_err is not None:
        raise last_err

    mp3 = out_dir / f"{source_id}.mp3"
    if not mp3.exists():
        # Fallback: pick newest file
        candidates = sorted(out_dir.glob(f"{source_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
        raise RuntimeError("Audio download did not produce an output file")
    return mp3


def _transcribe_audio_whisper(audio_path: Path, progress_cb: Callable[[], None] | None = None) -> str:
    # faster-whisper will download the model weights on first run.
    from faster_whisper import WhisperModel

    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), vad_filter=True, beam_size=1)
    parts: list[str] = []
    last_beat = time.time()
    for seg in segments:
        t = (getattr(seg, "text", "") or "").strip()
        if t:
            parts.append(t)
        if progress_cb is not None and (time.time() - last_beat) > 6:
            try:
                progress_cb()
            except Exception:
                pass
            last_beat = time.time()
    return "\n".join(parts).strip()


def _ingest_youtube_transcript(source_id: str, url: str) -> None:
    try:
        _update_source_fields(source_id, {"status": "processing", "progress": 8, "metadata": {"stage": "extracting_video_id"}})
        vid = _extract_youtube_id(url)
        if not vid:
            raise ValueError("Could not extract YouTube video id")

        _update_source_fields(source_id, {"status": "processing", "progress": 18, "metadata": {"stage": "fetching_transcript", "video_id": vid}})

        transcript_text = ""
        transcript_error: str | None = None
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            # Try direct transcript first.
            items = YouTubeTranscriptApi.get_transcript(vid)
            transcript_text = "\n".join(
                (it.get("text") or "").strip() for it in (items or []) if (it.get("text") or "").strip()
            )
        except Exception as e1:
            # Fallback: try transcript listing and select English/auto-generated.
            try:
                from youtube_transcript_api import YouTubeTranscriptApi

                tl = YouTubeTranscriptApi.list_transcripts(vid)
                # Prefer English if available; otherwise first transcript.
                t = None
                try:
                    t = tl.find_manually_created_transcript(["en", "en-US", "en-GB"])  # type: ignore[attr-defined]
                except Exception:
                    try:
                        t = tl.find_generated_transcript(["en", "en-US", "en-GB"])  # type: ignore[attr-defined]
                    except Exception:
                        # Take first available transcript
                        t = next(iter(tl), None)

                if t is None:
                    raise RuntimeError("No transcript tracks available")

                items = t.fetch()
                transcript_text = "\n".join(
                    (it.get("text") or "").strip() for it in (items or []) if (it.get("text") or "").strip()
                )
            except Exception as e2:
                transcript_error = f"{type(e1).__name__}: {e1}; fallback_failed={type(e2).__name__}: {e2}"

        # If transcript-api fails, fallback to yt-dlp subtitle extraction.
        if not (transcript_text or "").strip():
            _update_source_fields(
                source_id,
                {
                    "status": "processing",
                    "progress": 35,
                    "metadata": {"stage": "yt_dlp_subtitles", "video_id": vid, "note": "Trying yt-dlp fallback"},
                },
            )
            try:
                from yt_dlp import YoutubeDL

                subs_dir = TRANSCRIPTS_DIR / f"{source_id}_subs"
                subs_dir.mkdir(parents=True, exist_ok=True)

                ydl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "skip_download": True,
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitlesformat": "vtt",
                    "outtmpl": str(subs_dir / "%(id)s.%(ext)s"),
                }

                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    try:
                        ydl.download([url])
                    except Exception:
                        pass

                # Prefer English subtitles, otherwise take any available.
                def _pick_sub_url(info_dict: dict[str, Any]) -> tuple[str | None, str | None]:
                    subs = info_dict.get("subtitles") or {}
                    autos = info_dict.get("automatic_captions") or {}
                    for bucket in (subs, autos):
                        for lang in ("en", "en-US", "en-GB"):
                            tracks = bucket.get(lang) or []
                            for tr in tracks:
                                if str(tr.get("ext") or "").lower() == "vtt" and tr.get("url"):
                                    return str(tr.get("url")), lang
                    for bucket in (subs, autos):
                        for lang, tracks in bucket.items():
                            for tr in (tracks or []):
                                if str(tr.get("ext") or "").lower() == "vtt" and tr.get("url"):
                                    return str(tr.get("url")), str(lang)
                    return None, None

                sub_url, sub_lang = _pick_sub_url(info if isinstance(info, dict) else {})
                if not sub_url:
                    raise RuntimeError("yt-dlp: no subtitles/auto-captions available")

                _update_source_fields(
                    source_id,
                    {"status": "processing", "progress": 55, "metadata": {"stage": "downloading_vtt", "video_id": vid, "lang": sub_lang}},
                )
                vtt = ""
                try:
                    vtt_files = sorted(subs_dir.glob("*.vtt"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if vtt_files:
                        vtt = vtt_files[0].read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    vtt = ""
                if not vtt:
                    # Retry a few times in case of transient rate limits.
                    last_req_err: Exception | None = None
                    for attempt in range(3):
                        try:
                            r = requests.get(sub_url, timeout=30)
                            r.raise_for_status()
                            vtt = r.text
                            last_req_err = None
                            break
                        except Exception as e:
                            last_req_err = e
                            if _is_rate_limited_error(e):
                                time.sleep(2.0 + attempt * 3.0)
                                continue
                            break
                    if last_req_err is not None and not vtt:
                        raise last_req_err

                def _vtt_to_text(vtt_text: str) -> str:
                    lines = []
                    for ln in (vtt_text or "").splitlines():
                        t = ln.strip()
                        if not t:
                            continue
                        if t.upper() == "WEBVTT":
                            continue
                        if "-->" in t:
                            continue
                        if t.startswith("NOTE"):
                            continue
                        if re.match(r"^[0-9]+$", t):
                            continue
                        # Strip basic tags
                        t = re.sub(r"<[^>]+>", "", t)
                        lines.append(t)
                    # De-duplicate consecutive repeats
                    out_lines: list[str] = []
                    prev = None
                    for ln in lines:
                        if ln == prev:
                            continue
                        out_lines.append(ln)
                        prev = ln
                    return "\n".join(out_lines).strip()

                transcript_text = _vtt_to_text(vtt)
                if not transcript_text:
                    raise RuntimeError(f"yt-dlp: subtitles fetched but empty after parsing (len={len(vtt)})")
            except Exception as e3:
                transcript_error = (transcript_error or "") + f"; yt_dlp_failed={type(e3).__name__}: {e3}"

        # Final fallback: audio -> speech-to-text
        if not (transcript_text or "").strip():
            _update_source_fields(
                source_id,
                {
                    "status": "processing",
                    "progress": 60,
                    "metadata": {"stage": "audio_to_text", "video_id": vid, "note": "No captions; transcribing audio"},
                },
            )
            _ensure_ffmpeg_available()

            audio_dir = WORKSPACE_DIR / "audio"
            _update_source_fields(source_id, {"status": "processing", "progress": 65, "metadata": {"stage": "downloading_audio", "video_id": vid}})
            audio_path = _download_youtube_audio(url, audio_dir, source_id)

            # Surface the temporary audio artifact in metadata while processing.
            try:
                rel_audio = str(audio_path.relative_to(WORKSPACE_DIR)).replace("\\", "/")
            except Exception:
                rel_audio = str(audio_path)
            _update_source_fields(
                source_id,
                {
                    "status": "processing",
                    "progress": 72,
                    "metadata": {"stage": "audio_downloaded", "video_id": vid, "audio_path": rel_audio},
                },
            )

            _update_source_fields(source_id, {"status": "processing", "progress": 82, "metadata": {"stage": "transcribing", "video_id": vid}})

            # Trim long audio so transcription can't run unbounded.
            max_audio_s = _get_env_int("YT_MAX_AUDIO_S", 600)
            _update_source_fields(
                source_id,
                {"status": "processing", "progress": 78, "metadata": {"stage": "trimming_audio", "video_id": vid, "max_audio_s": max_audio_s}},
            )
            trimmed_audio = _trim_audio_for_transcription(audio_path, max_audio_s)

            beat = {"p": 82}
            def _beat() -> None:
                beat["p"] = min(89, int(beat["p"]) + 1)
                _update_source_fields(
                    source_id,
                    {"status": "processing", "progress": int(beat["p"]), "metadata": {"stage": "transcribing", "video_id": vid}},
                )

            _update_source_fields(source_id, {"status": "processing", "progress": 82, "metadata": {"stage": "transcribing", "video_id": vid}})
            transcript_text = _transcribe_audio_whisper(trimmed_audio, progress_cb=_beat)
            if not (transcript_text or "").strip():
                raise RuntimeError("Audio transcription produced empty text")

            # Delete the temporary audio file once transcript exists.
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                if trimmed_audio != audio_path:
                    trimmed_audio.unlink(missing_ok=True)
            except Exception:
                pass

        transcript_text = (transcript_text or "").strip()
        if not transcript_text:
            raise RuntimeError(transcript_error or "Transcript is empty or unavailable")

        _update_source_fields(source_id, {"status": "processing", "progress": 70, "metadata": {"stage": "writing_transcript", "video_id": vid}})
        transcript_path = TRANSCRIPTS_DIR / f"{source_id}.txt"
        transcript_path.write_text(transcript_text, encoding="utf-8", errors="ignore")

        meta = {"video_id": vid, "chars": len(transcript_text)}
        _update_source_fields(source_id, {"status": "ready", "progress": 100, "metadata": meta})
    except Exception as e:
        # Keep a short error file for debugging, but mark the source as error.
        try:
            tp = TRANSCRIPTS_DIR / f"{source_id}.txt"
            if not tp.exists():
                tp.write_text(f"URL: {url}\n\nERROR: {e}\n", encoding="utf-8", errors="ignore")
        except Exception:
            pass
        _update_source_fields(
            source_id,
            {
                "status": "error",
                "progress": None,
                "metadata": {
                    "error": str(e),
                    "url": url,
                    "hint": "If this is a YouTube video without captions, install ffmpeg and ensure faster-whisper is installed for audio-to-text transcription.",
                },
            },
        )


def _upsert_source(source: dict[str, Any]) -> None:
    sources = _read_sources()
    sid = str(source.get("id") or "")
    if not sid:
        source["id"] = str(int(time.time() * 1000))
        sources.append(source)
        _write_sources(sources)
        return

    for i, s in enumerate(sources):
        if str(s.get("id")) == sid:
            sources[i] = {**s, **source}
            _write_sources(sources)
            return

    sources.append(source)
    _write_sources(sources)


def _delete_sources(filename: str | None = None) -> None:
    if filename is None:
        _write_sources([])
        return

    sources = _read_sources()
    filename_l = filename.strip().lower()
    next_sources = [s for s in sources if str(s.get("name", "")).strip().lower() != filename_l]
    _write_sources(next_sources)


class AskRequest(BaseModel):
    query: str
    top_k: int = 4
    include_youtube: bool = False
    history: list[dict[str, Any]] = Field(default_factory=list)


class SimpleGenRequest(BaseModel):
    query: str | None = None
    top_k: int = 4
    include_youtube: bool = False


class SummaryRequest(BaseModel):
    mode: Literal["concise", "detailed"] = "detailed"
    max_chars: int = 2000


class YoutubeRequest(BaseModel):
    url: str


class LogQuizRequest(BaseModel):
    quiz_data: list[dict[str, Any]]
    user_answers: dict[int, str]


class LogFlashcardRequest(BaseModel):
    flashcard_data: list[dict[str, Any]]
    ratings: dict[int, int]


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


# NVIDIA-only implementation - no external LLM fallbacks
def _llm_generate(prompt: str) -> str:
    """Generate completion using NVIDIA NIM API only."""
    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    model = (os.getenv("NVIDIA_MODEL") or "abacusai/dracarys-llama-3.1-70b-instruct").strip()
    timeout_s = _get_env_int("NVIDIA_TIMEOUT_S", 60)

    if not api_key:
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY not configured")

    try:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount("https://", adapter)
        
        r = session.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 2000,
                "stream": False,
            },
            timeout=(10, timeout_s),
        )
        
        if r.status_code >= 400:
            error_body = (r.text or "")[:200]
            raise HTTPException(
                status_code=502,
                detail=f"NVIDIA API error (status={r.status_code}): {error_body}",
            )

        data = r.json()
        content = (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content")
        if content is None:
            raise HTTPException(status_code=502, detail="NVIDIA API returned empty response")
        return str(content)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NVIDIA request failed: {type(e).__name__}: {e}")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {"ok": True, "service": "knowledge_workspace_backend"}


@app.get("/config")
def config() -> dict[str, Any]:
    has_key = bool((os.getenv("NVIDIA_API_KEY") or "").strip())
    
    return {
        "provider": "nvidia" if has_key else None,
        "model": (os.getenv("NVIDIA_MODEL") or "abacusai/dracarys-llama-3.1-70b-instruct").strip(),
        "timeout_s": _get_env_int("NVIDIA_TIMEOUT_S", 60),
        "has_api_key": has_key,
    }


@app.post("/reset")
def reset() -> dict[str, Any]:
    _write_sources([])
    for p in UPLOADS_DIR.glob("*"):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass
    for p in TRANSCRIPTS_DIR.glob("*"):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    name = Path(file.filename).name
    dest = UPLOADS_DIR / f"{int(time.time() * 1000)}_{name}"

    content = await file.read()
    dest.write_bytes(content)

    # Extract text using the new ingestion system
    ingester = create_document_ingester()
    extract_result = ingester.extract_text_and_metadata(dest, name)
    
    # Prepare source metadata
    metadata = {
        "bytes": len(content),
        "extracted": extract_result.success,
        "extraction_metadata": extract_result.metadata
    }
    
    # Add error message if extraction failed
    if not extract_result.success:
        metadata["unreadable"] = True
        metadata["extraction_error"] = extract_result.error_message
    else:
        metadata["text_length"] = len(extract_result.text)
    
    # Store extracted text in metadata for immediate use
    if extract_result.success and extract_result.text.strip():
        metadata["extracted_text"] = extract_result.text[:10000]  # Store first 10k chars
    
    src = {
        "id": str(int(time.time() * 1000)),
        "type": "document",
        "name": name,
        "path": str(dest.relative_to(WORKSPACE_DIR)).replace("\\", "/"),
        "status": "ready",
        "metadata": metadata,
    }
    _upsert_source(src)

    return {"ok": True, "stats": src.get("metadata")}


@app.post("/youtube")
def youtube(req: YoutubeRequest) -> dict[str, Any]:
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")

    sid = str(int(time.time() * 1000))
    src = {
        "id": sid,
        "type": "youtube",
        "name": url,
        "url": url,
        "status": "processing",
        "progress": 0,
        "metadata": {"stage": "queued"},
    }
    _upsert_source(src)

    t = threading.Thread(target=_ingest_youtube_transcript, args=(sid, url), daemon=True)
    t.start()

    return {"ok": True, "id": sid, "stats": src.get("metadata")}


@app.get("/sources")
def get_sources() -> dict[str, Any]:
    sources = _read_sources()
    return {"sources": sources}


@app.delete("/sources")
def delete_sources(filename: str | None = None) -> dict[str, Any]:
    _delete_sources(filename)
    return {"ok": True}


def _current_citations() -> list[dict[str, Any]]:
    sources = _read_sources()
    citations: list[dict[str, Any]] = []
    for s in sources[:8]:
        content_preview = ""
        try:
            if s.get("type") == "document" and s.get("path"):
                p = WORKSPACE_DIR / str(s.get("path"))
                if p.exists() and p.is_file():
                    # Check if we have pre-extracted text in metadata
                    metadata = s.get("metadata", {})
                    if metadata.get("extracted") and metadata.get("extracted_text"):
                        content_preview = metadata["extracted_text"][:5000]
                    else:
                        # Fall back to on-demand extraction for backward compatibility
                        if p.suffix.lower() == ".docx":
                            import docx
                            doc = docx.Document(p)
                            content_preview = "\n".join(par.text for par in doc.paragraphs)[:5000]
                        else:
                            raw = p.read_bytes()[:5000]
                            content_preview = raw.decode("utf-8", errors="ignore")
            elif s.get("type") == "youtube" and s.get("id"):
                tp = TRANSCRIPTS_DIR / f"{s.get('id')}.txt"
                if tp.exists() and tp.is_file():
                    content_preview = tp.read_text(encoding="utf-8", errors="ignore")[:5000]
        except Exception:
            content_preview = ""

        md = dict(s.get("metadata") or {})
        if "filename" not in md and s.get("name"):
            md["filename"] = s.get("name")
        if "url" not in md and s.get("url"):
            md["url"] = s.get("url")

        citations.append(
            {
                "id": s.get("id"),
                "type": s.get("type"),
                "name": s.get("name"),
                "url": s.get("url"),
                "content": content_preview,
                "metadata": md,
            }
        )
    return citations


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    sources = _current_citations()

    context = _build_context_for_llm(sources)
    if context:
        prompt = (
            "You are a friendly, natural conversational assistant. Answer like a human tutor. "
            "Use ONLY the provided document content as factual grounding. "
            "Do not say 'according to the source' or mention citations. "
            "If the answer is not in the documents, say you don't have enough information in the uploaded content.\n\n"
            f"DOCUMENTS:\n{context}\n\n"
            f"USER QUESTION: {query}\n"
        )
    else:
        prompt = (
            "You are a friendly, natural conversational assistant. Answer like a human. "
            "If the user expects you to use uploaded documents, ask them to upload content first.\n\n"
            f"USER QUESTION: {query}\n"
        )

    answer = _llm_generate(prompt)

    return {"answer": answer, "sources": sources}


@app.post("/summary")
def summary(req: SummaryRequest) -> dict[str, Any]:
    sources = _current_citations()

    context = _build_context_for_llm(sources)
    if context:
        prompt = (
            "Write a crisp, easy-to-read summary in 1–2 short paragraphs. "
            "Treat it like real content; do not mention 'sources' or filenames. "
            "Use plain language and include the main sections/topics covered.\n\n"
            f"DOCUMENTS:\n{context}\n"
        )
    else:
        prompt = "No documents were provided. Reply with a single sentence asking the user to upload a document first."

    out = _llm_generate(prompt)

    max_chars = max(50, int(req.max_chars))
    return {"summary": out[:max_chars], "sources": sources}


@app.post("/quiz")
def quiz(req: SimpleGenRequest) -> dict[str, Any]:
    sources = _current_citations()
    if not sources:
        raise HTTPException(status_code=400, detail="No sources")

    context = _build_context_for_llm(sources)
    prompt = (
        "Create a quiz of 5–10 multiple-choice questions based on the document content. "
        "Return ONLY valid JSON with this exact shape: "
        "{\"quiz\":[{\"question\":string,\"options\":[string,string,string,string],\"answer\":string,\"explanation\":string}]} "
        "Rules: options must be realistic; answer must exactly match one of the options; keep questions relevant to the content.\n\n"
        f"DOCUMENTS:\n{context}\n"
    )
    out = _llm_generate(prompt)
    parsed = _safe_json_loads(out or "")
    quiz_items = None
    if isinstance(parsed, dict) and isinstance(parsed.get("quiz"), list):
        quiz_items = parsed.get("quiz")
    if not isinstance(quiz_items, list) or not quiz_items:
        raise HTTPException(status_code=502, detail="NVIDIA NIM returned invalid quiz JSON")

    return {"quiz": quiz_items, "sources": sources}


@app.post("/flashcards")
def flashcards(req: SimpleGenRequest) -> dict[str, Any]:
    sources = _current_citations()
    if not sources:
        raise HTTPException(status_code=400, detail="No sources")

    context = _build_context_for_llm(sources)
    prompt = (
        "Create 8–12 study flashcards based on the document. "
        "Return ONLY valid JSON with this exact shape: "
        "{\"flashcards\":[{\"front\":string,\"back\":string}]} "
        "Rules: front is a short prompt/term/question; back is the explanation/answer.\n\n"
        f"DOCUMENTS:\n{context}\n"
    )
    out = _llm_generate(prompt)
    parsed = _safe_json_loads(out or "")
    cards = None
    if isinstance(parsed, dict) and isinstance(parsed.get("flashcards"), list):
        cards = parsed.get("flashcards")
    if not isinstance(cards, list) or not cards:
        raise HTTPException(status_code=502, detail="NVIDIA NIM returned invalid flashcards JSON")

    return {"flashcards": cards, "sources": sources}


@app.post("/mindmap")
def mindmap(req: SimpleGenRequest) -> dict[str, Any]:
    sources = _current_citations()
    if not sources:
        raise HTTPException(status_code=400, detail="No sources")

    context = _build_context_for_llm(sources)
    prompt = (
        "Create a mind map based on the document. Return ONLY valid JSON with this exact shape: "
        "{\"mindmap\":{\"nodes\":[{\"id\":string,\"label\":string}],\"edges\":[{\"from\":string,\"to\":string}]}}. "
        "Rules: the first node must be the main topic; connect edges meaningfully; keep it under 18 nodes.\n\n"
        f"DOCUMENTS:\n{context}\n"
    )
    out = _llm_generate(prompt)
    parsed = _safe_json_loads(out or "")

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    if isinstance(parsed, dict) and isinstance(parsed.get("mindmap"), dict):
        mm = parsed.get("mindmap")
        if isinstance(mm.get("nodes"), list):
            nodes = [n for n in mm.get("nodes") if isinstance(n, dict) and n.get("id") and n.get("label")]
        if isinstance(mm.get("edges"), list):
            edges = [e for e in mm.get("edges") if isinstance(e, dict) and e.get("from") and e.get("to")]

    if not nodes:
        raise HTTPException(status_code=502, detail="NVIDIA NIM returned invalid mindmap JSON")
    if not edges:
        # Minimal meaningful connections: connect root to all other nodes.
        root_id = str(nodes[0].get("id"))
        for n in nodes[1:]:
            edges.append({"from": root_id, "to": str(n.get("id"))})

    return {"mindmap": {"nodes": nodes, "edges": edges}, "sources": sources}


@app.post("/infographic")
def infographic(req: SimpleGenRequest) -> dict[str, Any]:
    sources = _current_citations()
    if not sources:
        raise HTTPException(status_code=400, detail="No sources")

    context = _build_context_for_llm(sources)
    prompt = (
        "Create an infographic from the document. Return ONLY valid JSON with this exact shape: "
        "{\"title\":string,\"items\":[{\"heading\":string,\"text\":string,\"details\":string}]}. "
        "Rules: 5–8 items; heading is short; text is 1 sentence; details is 2–5 sentences expanding on it.\n\n"
        f"DOCUMENTS:\n{context}\n"
    )
    out = _llm_generate(prompt)
    parsed = _safe_json_loads(out or "")
    info = None
    if isinstance(parsed, dict) and isinstance(parsed.get("title"), str) and isinstance(parsed.get("items"), list):
        info = {"title": parsed.get("title"), "items": parsed.get("items")}
    if info is None:
        raise HTTPException(status_code=502, detail="NVIDIA NIM returned invalid infographic JSON")
    return {"infographic": info, "sources": sources}


# Learner Model Endpoints for Cognitive Twin Feature

@app.get("/learner-model")
def get_learner_model() -> dict[str, Any]:
    """Get the complete learner model data"""
    try:
        manager = LearnerModelManager(WORKSPACE_DIR)
        model = manager.load_learner_model()
        return {"learner_model": model.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load learner model: {str(e)}")


@app.post("/learner-model/log-quiz")
def log_quiz_results(req: LogQuizRequest) -> dict[str, Any]:
    """Log quiz completion results and update learner model"""
    try:
        manager = LearnerModelManager(WORKSPACE_DIR)
        model = manager.load_learner_model()
        
        manager.log_quiz_results(model, req.quiz_data, req.user_answers)
        manager.save_learner_model(model)
        
        return {"ok": True, "concepts_updated": len(model.concepts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log quiz results: {str(e)}")


@app.post("/learner-model/log-flashcard")
def log_flashcard_results(req: LogFlashcardRequest) -> dict[str, Any]:
    """Log flashcard study results and update learner model"""
    try:
        manager = LearnerModelManager(WORKSPACE_DIR)
        model = manager.load_learner_model()
        
        manager.log_flashcard_results(model, req.flashcard_data, req.ratings)
        manager.save_learner_model(model)
        
        return {"ok": True, "concepts_updated": len(model.concepts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log flashcard results: {str(e)}")


@app.get("/learner-model/recommendations")
def get_recommendations() -> dict[str, Any]:
    """Get daily study recommendations"""
    try:
        manager = LearnerModelManager(WORKSPACE_DIR)
        model = manager.load_learner_model()
        
        # Generate fresh recommendations
        recommendations = manager.generate_recommendations(model)
        
        return {"recommendations": [r.dict() for r in recommendations]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


# Curriculum Builder Endpoints for Knowledge Curriculum Architect

@app.post("/curriculum/build")
def build_curriculum(req: BuildCurriculumRequest) -> dict[str, Any]:
    """Build or rebuild curriculum from sources and learner data"""
    try:
        # Initialize curriculum builder
        builder = CurriculumBuilder(WORKSPACE_DIR)
        
        # Build curriculum
        curriculum = builder.build_curriculum(force=req.force, llm_generate_func=_llm_generate)
        
        return {
            "curriculum": curriculum.dict(),
            "built_at": curriculum.metadata.get("built_at", ""),
            "personalized": curriculum.metadata.get("personalized", False)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Curriculum building failed: {str(e)}")


@app.get("/curriculum")
def get_curriculum() -> dict[str, Any]:
    """Get existing curriculum or build one if none exists"""
    try:
        builder = CurriculumBuilder(WORKSPACE_DIR)
        
        # Try to load existing curriculum
        curriculum = builder.load_curriculum()
        
        if curriculum is None:
            # Build curriculum if none exists
            curriculum = builder.build_curriculum(force=False, llm_generate_func=_llm_generate)
        
        return {
            "curriculum": curriculum.dict(),
            "built_at": curriculum.metadata.get("built_at", ""),
            "personalized": curriculum.metadata.get("personalized", False)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load or build curriculum: {str(e)}")


# Comparison Engine Endpoints for Contradiction & Consensus Explorer

@app.post("/compare/sources")
def compare_sources(req: CompareRequest) -> dict[str, Any]:
    """Compare multiple sources to find agreements, contradictions, and unique points"""
    try:
        # Validate source IDs
        if not req.source_ids:
            raise HTTPException(status_code=400, detail="No source IDs provided")
        
        if len(req.source_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 sources required for comparison")
        
        # Initialize comparison engine
        engine = ComparisonEngine(WORKSPACE_DIR)
        
        # Run analysis
        result = engine.analyze_sources(req.source_ids, req.query, _llm_generate)
        
        return {
            "comparison": result.dict(),
            "sources_analyzed": len(req.source_ids),
            "query": req.query
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison analysis failed: {str(e)}")


# Initialize recommendation import handler
_import_handler: ImportHandler | None = None

def _get_import_handler() -> ImportHandler:
    """Get or create the import handler with dependencies"""
    global _import_handler
    if _import_handler is None:
        _import_handler = create_import_handler(
            workspace_dir=WORKSPACE_DIR,
            read_sources_func=_read_sources,
            write_sources_func=_write_sources,
            upsert_source_func=_upsert_source,
            youtube_ingest_func=_ingest_youtube_transcript,
            document_ingest_func=lambda path, name: create_document_ingester().extract_text_and_metadata(path, name).to_dict(),
        )
    return _import_handler


@app.post("/recommend")
def recommend(filters: RecommendationFilters) -> dict[str, Any]:
    """Get learning resource recommendations based on topic and filters"""
    try:
        engine = RecommendationEngine(WORKSPACE_DIR)
        result = engine.get_recommendations(filters)
        
        return {
            "ok": True,
            "items": [item.dict() for item in result.items],
            "total_found": result.total_found,
            "search_topic": result.search_topic,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@app.post("/recommend/from-source")
def recommend_from_source(source_id: str, filters: RecommendationFilters) -> dict[str, Any]:
    """Get recommendations based on content of an uploaded source (PDF/YouTube)"""
    try:
        engine = RecommendationEngine(WORKSPACE_DIR)
        result = engine.get_recommendations_for_source(source_id, filters)
        
        return {
            "ok": True,
            "items": [item.dict() for item in result.items],
            "total_found": result.total_found,
            "search_topic": result.search_topic,
            "extracted_topics": getattr(result, 'extracted_topics', []),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze source: {str(e)}")


@app.post("/recommend/import")
def import_recommendation(req: ImportRequest) -> dict[str, Any]:
    """Import a recommended resource into the workspace"""
    try:
        handler = _get_import_handler()
        result = handler.import_resource(req)
        
        return {
            "ok": result.ok,
            "source_id": result.source_id,
            "message": result.message,
            "import_type": result.import_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


if __name__ == "__main__":
    host = (os.getenv("HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = _get_env_int("PORT", 8000)
    uvicorn.run("app:app", host=host, port=port, reload=False)
