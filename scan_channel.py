import os
import re
import csv
import time
import glob
import shutil
import tempfile
import yaml
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
import google.generativeai as genai

try:
  import yt_dlp
except Exception:
  yt_dlp = None

# === CONFIG ===
CONFIG_PATH = "config.yml"
with open(CONFIG_PATH, "r", encoding="utf-8") as cf:
  cfg = yaml.safe_load(cf) or {}

API_KEYS_PATH = "api_keys.yml"
try:
  with open(API_KEYS_PATH, "r", encoding="utf-8") as kf:
    keys = yaml.safe_load(kf) or {}
except Exception:
  keys = {}

CHANNEL_URL     = cfg["youtube"]["channel_url"]
OUTPUT_FILE     = cfg["run"]["output_file"]
MAX_VIDEOS      = int(cfg["run"]["max_videos"])
SKIP_PROCESSED  = bool(cfg["run"]["skip_processed"])
POLITE_DELAY    = int(cfg["run"]["polite_delay_sec"]) 
DEBUG_MODE      = bool(cfg["run"]["debug"]) 
USE_YTDLP       = bool(cfg["run"]["use_ytdlp_fallback"]) 
PROMPT_TEMPLATE = cfg["run"]["prompt"]

LLM_PROVIDER = cfg["llm"]["provider"].lower()
LLM_MODEL    = cfg["llm"]["model"]
_llm_key_env    = os.getenv("YT_SCANNER_API_KEY")
_llm_key_file   = keys.get("llm") if isinstance(keys, dict) else None
_llm_key_legacy = cfg.get("llm", {}).get("api_key")
LLM_KEY         = _llm_key_env or _llm_key_file or _llm_key_legacy

# === INIT ===
if LLM_PROVIDER == "gemini":
  if not LLM_KEY:
    raise SystemExit("❌ Missing API key. Set env YT_SCANNER_API_KEY or api_keys.yml key 'llm' (or legacy config.llm.api_key)")
  genai.configure(api_key=LLM_KEY)
  model = genai.GenerativeModel(LLM_MODEL)
else:
  raise SystemExit(f"❌ Unsupported LLM provider: {LLM_PROVIDER}")

# === FUNCTIONS ===
def fetch_video_ids_via_ytdlp(limit=100):
  """Use yt-dlp to extract video IDs from channel. Requires yt-dlp."""
  if yt_dlp is None:
    if DEBUG_MODE:
      print("[debug] yt-dlp not installed; cannot use fallback")
    return []
  urls_to_try = [CHANNEL_URL]
  # try both handle page and without /videos
  if CHANNEL_URL.endswith('/videos'):
    urls_to_try.append(CHANNEL_URL.rsplit('/videos', 1)[0])
  ydl_opts = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': 'in_playlist',
    'noplaylist': False,
    'playlistend': limit,
    'extractor_args': {'youtube': {'lang': ['en']}}
  }
  ids = []
  seen = set()
  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for u in urls_to_try:
      try:
        info = ydl.extract_info(u, download=False)
        entries = info.get('entries') or []
        for e in entries:
          vid = (e.get('id') or '').strip()
          if vid and vid not in seen:
            ids.append(vid)
            seen.add(vid)
          if len(ids) >= limit:
            break
        if DEBUG_MODE:
          print(f"[debug] yt-dlp extracted {len(entries)} entries from {u}; collected {len(ids)} unique IDs")
        if ids:
          break
      except Exception as e:
        if DEBUG_MODE:
          print(f"[debug] yt-dlp extract failed for {u}: {e}")
        continue
  return ids[:limit]

def get_transcript(video_id):
  """Fetch transcript text with best effort (manual/en, generated/en, or translate to en), with retries."""
  attempts = 3
  delay = 2
  for attempt in range(1, attempts + 1):
    try:
      transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

      # 1) Prefer manually created English
      try:
        t = transcripts.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        return " ".join([x['text'] for x in t.fetch()])
      except Exception:
        pass

      # 2) Prefer generated English
      try:
        t = transcripts.find_generated_transcript(['en', 'en-US', 'en-GB'])
        return " ".join([x['text'] for x in t.fetch()])
      except Exception:
        pass

      # 3) Any available transcript (any language)
      for t in transcripts:
        try:
          return " ".join([x['text'] for x in t.fetch()])
        except Exception:
          continue

      # 4) Try any transcript translated to English
      for t in transcripts:
        try:
          tt = t.translate('en')
          return " ".join([x['text'] for x in tt.fetch()])
        except Exception:
          continue

      # Try yt-dlp fallback for subtitles if enabled
      if USE_YTDLP:
        text = get_transcript_via_ytdlp(video_id)
        if text:
          return text
      return None
    except (TranscriptsDisabled, NoTranscriptFound):
      if USE_YTDLP:
        text = get_transcript_via_ytdlp(video_id)
        if text:
          return text
      return None
    except CouldNotRetrieveTranscript:
      if attempt < attempts:
        time.sleep(delay)
        delay *= 2
        continue
      if USE_YTDLP:
        text = get_transcript_via_ytdlp(video_id)
        if text:
          return text
      return None
    except Exception:
      if attempt < attempts:
        time.sleep(delay)
        delay *= 2
        continue
      if USE_YTDLP:
        text = get_transcript_via_ytdlp(video_id)
        if text:
          return text
      return None

def get_transcript_via_ytdlp(video_id):
  """Use yt-dlp to fetch English subtitles (.vtt) and return plain text, if available."""
  if yt_dlp is None:
    return None
  url = f"https://www.youtube.com/watch?v={video_id}"
  tmpdir = tempfile.mkdtemp(prefix="yt_subs_")
  try:
    # Try manual English subs first
    opts_manual = {
      'quiet': True,
      'skip_download': True,
      'writesubtitles': True,
      'subtitleslangs': ['en'],
      'subtitlesformat': 'vtt',
      'outtmpl': f"{tmpdir}/%(id)s.%(ext)s"
    }
    with yt_dlp.YoutubeDL(opts_manual) as ydl:
      try:
        ydl.download([url])
      except Exception:
        pass
    vtts = glob.glob(os.path.join(tmpdir, f"{video_id}*.vtt"))
    if not vtts:
      # Try auto English subs
      opts_auto = {
        'quiet': True,
        'skip_download': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'outtmpl': f"{tmpdir}/%(id)s.%(ext)s"
      }
      with yt_dlp.YoutubeDL(opts_auto) as ydl:
        try:
          ydl.download([url])
        except Exception:
          pass
      vtts = glob.glob(os.path.join(tmpdir, f"{video_id}*.vtt"))
    if not vtts:
      return None
    # Read and strip VTT cues
    texts = []
    for fp in vtts:
      try:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
          for line in f:
            s = line.strip()
            if not s:
              continue
            if s.startswith('WEBVTT') or '-->' in s or s.isdigit():
              continue
            texts.append(s)
      except Exception:
        continue
    return " ".join(texts).strip() or None
  finally:
    try:
      shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
      pass

def extract_business_idea(transcript):
  """Extract short business idea (2–8 words) using Gemini."""
  if "{transcript}" in PROMPT_TEMPLATE:
    prompt = PROMPT_TEMPLATE.replace("{transcript}", transcript)
  else:
    prompt = f"{PROMPT_TEMPLATE}\nTranscript:\n\n{transcript}"
  try:
    response = model.generate_content(prompt)
    return response.text.strip().replace("\n", " ")
  except Exception as e:
    print("⚠️ Gemini error:", e)
    return None

# === MAIN ===
def main():
  # Prefer yt-dlp discovery if enabled
  video_ids = []
  if USE_YTDLP:
    if yt_dlp is None and DEBUG_MODE:
      print("[debug] yt-dlp is not installed; cannot use yt-dlp discovery")
    else:
      if DEBUG_MODE:
        print("[debug] Trying yt-dlp discovery first…")
      video_ids = fetch_video_ids_via_ytdlp(MAX_VIDEOS)
      if DEBUG_MODE:
        print(f"[debug] yt-dlp discovery collected {len(video_ids)} IDs")
  if not video_ids:
    print("🔎 Found 0 videos via yt-dlp. Enable run.debug for details.")
    print("❌ No videos to process. Exiting.")
    return

  processed_urls = set()
  if SKIP_PROCESSED and os.path.exists(OUTPUT_FILE):
    try:
      with open(OUTPUT_FILE, "r", newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf)
        next(reader, None)
        for row in reader:
          if row:
            processed_urls.add(row[0])
    except Exception:
      processed_urls = set()

  file_exists = os.path.exists(OUTPUT_FILE)
  file_empty = False
  if file_exists:
    try:
      file_empty = os.path.getsize(OUTPUT_FILE) == 0
    except Exception:
      file_empty = False

  with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists or file_empty:
      writer.writerow(["Video URL", "Summary"])

    total = len(video_ids)
    for i, vid in enumerate(video_ids, 1):
      url = f"https://www.youtube.com/watch?v={vid}"
      if url in processed_urls:
        print(f"\n[{i}/{total}] Skipping already processed {url}")
        continue

      print(f"\n[{i}/{total}] Processing {url}")
      transcript = get_transcript(vid)
      if not transcript:
        print("  ❌ No transcript found, writing n/a.")
        writer.writerow([url, "n/a"])  # mark as processed with n/a
        time.sleep(POLITE_DELAY)
        continue

      idea = extract_business_idea(transcript)
      if not idea:
        print("  ⚠️ No idea extracted, writing n/a.")
        writer.writerow([url, "n/a"])  # mark as processed with n/a
        time.sleep(POLITE_DELAY)
        continue

      print(f"  ✅ Idea: {idea}")
      writer.writerow([url, idea])

      time.sleep(POLITE_DELAY)

  print(f"\n✅ Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
  main()
