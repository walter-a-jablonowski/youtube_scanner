import os
import re
import csv
import time
import glob
import shutil
import tempfile
import yaml
import sys
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

DEBUG_MODE      = bool(cfg["run"]["debug"]) 
USE_YTDLP       = bool(cfg["run"]["use_ytdlp_fallback"]) 

# Globals configured per-task at runtime
PROMPT_TEMPLATE = None
OUTPUT_FILE     = None
MAX_VIDEOS      = None
SKIP_PROCESSED  = None
POLITE_DELAY    = None

def load_task_file(task_path):
  try:
    with open(task_path, "r", encoding="utf-8") as tf:
      return yaml.safe_load(tf) or {}
  except Exception as e:
    if DEBUG_MODE:
      print(f"[debug] Error loading task file: {e}")
    return None

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
def fetch_video_ids_via_ytdlp(limit=100, channel_url=None, channel_label=None):
  """Use yt-dlp to extract videos with minimal metadata. Requires yt-dlp.

  Returns a list of dicts: {id, upload_date, timestamp, channel}
  """
  if yt_dlp is None:
    if DEBUG_MODE:
      print("[debug] yt-dlp not installed; cannot use fallback")
    return []
  base_url = channel_url
  urls_to_try = [base_url]
  # try both handle page and without /videos
  if base_url.endswith('/videos'):
    urls_to_try.append(base_url.rsplit('/videos', 1)[0])
  ydl_opts = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': 'in_playlist',
    'noplaylist': False,
    'playlistend': limit,
    'extractor_args': {'youtube': {'lang': ['en']}}
  }
  items = []
  seen = set()
  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for u in urls_to_try:
      try:
        info = ydl.extract_info(u, download=False)
        entries = info.get('entries') or []
        for e in entries:
          vid = (e.get('id') or '').strip()
          if not vid or vid in seen:
            continue
          seen.add(vid)
          # Metadata may vary in flat mode
          upload_date = e.get('upload_date')  # YYYYMMDD if present
          ts = e.get('timestamp') or e.get('release_timestamp')
          channel_name = e.get('channel') or e.get('uploader')
          item = {
            'id': vid,
            'upload_date': upload_date,
            'timestamp': ts,
            'channel': channel_name,
          }
          # Enrich missing fields with a per-video metadata call
          if not (item['timestamp'] or item['upload_date']) or not item['channel']:
            try:
              vinfo = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
              item['upload_date'] = item['upload_date'] or vinfo.get('upload_date')
              item['timestamp']   = item['timestamp'] or vinfo.get('timestamp') or vinfo.get('release_timestamp')
              item['channel']     = item['channel'] or vinfo.get('channel') or vinfo.get('uploader')
            except Exception as ve:
              if DEBUG_MODE:
                print(f"[debug] enrich failed for {vid}: {ve}")
              pass
          if not item['channel'] and channel_label:
            item['channel'] = channel_label
          items.append(item)
          if len(items) >= limit:
            break
        if DEBUG_MODE:
          print(f"[debug] yt-dlp extracted {len(entries)} entries from {u}; collected {len(items)} unique IDs")
        if items:
          break
      except Exception as e:
        if DEBUG_MODE:
          print(f"[debug] yt-dlp extract failed for {u}: {e}")
        continue
  return items[:limit]

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

def extract_summary(transcript):
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
def process_task(task, task_name):
  global PROMPT_TEMPLATE, OUTPUT_FILE, MAX_VIDEOS, SKIP_PROCESSED, POLITE_DELAY
  # Configure per-task settings (required)
  PROMPT_TEMPLATE = task["prompt"]
  OUTPUT_FILE     = task["output_file"]
  MAX_VIDEOS      = int(task["max_videos"]) 
  SKIP_PROCESSED  = bool(task["skip_processed"]) 
  POLITE_DELAY    = int(task["polite_delay_sec"]) 

  # Build channels from task
  CHANNELS = []  # list of (label, url)
  for label, data in (task.get("channels") or {}).items():
    url = (data or {}).get("url")
    if url:
      CHANNELS.append((label, url))
  if not CHANNELS:
    print(f"❌ Task '{task_name}' has no channels configured.")
    return

  # Discover videos across channels
  all_items = []
  if USE_YTDLP:
    if yt_dlp is None and DEBUG_MODE:
      print("[debug] yt-dlp is not installed; cannot use yt-dlp discovery")
    else:
      for label, url in CHANNELS:
        if DEBUG_MODE:
          print(f"[debug] Discovering channel {label}: {url}")
        items = fetch_video_ids_via_ytdlp(MAX_VIDEOS, channel_url=url, channel_label=label)
        if DEBUG_MODE:
          print(f"[debug] {label}: collected {len(items)} IDs")
        all_items.extend(items)
  if not all_items:
    print("🔎 Found 0 videos via yt-dlp. Enable run.debug for details.")
    print(f"❌ No videos to process for task '{task_name}'.")
    return

  # Read existing processed URLs
  processed_urls = set()
  if SKIP_PROCESSED and os.path.exists(OUTPUT_FILE):
    try:
      with open(OUTPUT_FILE, "r", newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf)
        next(reader, None)
        for row in reader:
          if row:
            processed_urls.add(row[3])  # Video URL column
    except Exception:
      processed_urls = set()

  # Header setup
  file_exists = os.path.exists(OUTPUT_FILE)
  file_empty = False
  if file_exists:
    try:
      file_empty = os.path.getsize(OUTPUT_FILE) == 0
    except Exception:
      file_empty = False

  def _format_time(item):
    try:
      if item.get('timestamp'):
        return time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(item['timestamp'])))
      d = item.get('upload_date')
      if d and len(d) == 8:
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    except Exception:
      pass
    return ""

  # Ensure output directory exists
  out_dir = os.path.dirname(OUTPUT_FILE)
  if out_dir:
    try:
      os.makedirs(out_dir, exist_ok=True)
    except Exception:
      pass

  with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists or file_empty:
      writer.writerow(["Time", "Channel", "Summary", "Video URL"])

    total = len(all_items)
    for i, item in enumerate(all_items, 1):
      vid = item['id']
      url = f"https://www.youtube.com/watch?v={vid}"
      if url in processed_urls:
        print(f"\n[{i}/{total}] Skipping already processed {url}")
        continue

      print(f"\n[{i}/{total}] Processing {url}")
      transcript = get_transcript(vid)
      if not transcript:
        print("  ❌ No transcript found, writing n/a.")
        time_str = _format_time(item)
        channel_name = item.get('channel') or ""
        writer.writerow([time_str, channel_name, "n/a", url])
        time.sleep(POLITE_DELAY)
        continue

      idea = extract_summary(transcript)
      if not idea:
        print("  ⚠️ No idea extracted, writing n/a.")
        time_str = _format_time(item)
        channel_name = item.get('channel') or ""
        writer.writerow([time_str, channel_name, "n/a", url])
        time.sleep(POLITE_DELAY)
        continue

      print(f"  ✅ Idea: {idea}")
      time_str = _format_time(item)
      channel_name = item.get('channel') or ""
      writer.writerow([time_str, channel_name, idea, url])

      time.sleep(POLITE_DELAY)

  print(f"\n✅ Done! Saved to {OUTPUT_FILE}")

def main():
  # CLI: python scan_channel.py ALL | task_name (basename without .yml)
  arg = sys.argv[1] if len(sys.argv) > 1 else None
  tasks_dir = os.path.join("tasks")
  if not os.path.isdir(tasks_dir):
    print("❌ tasks/ directory not found")
    return

  task_files = []
  if arg and arg.upper() == "ALL":
    for name in os.listdir(tasks_dir):
      if name.lower().endswith('.yml') and not name.startswith('.'):
        task_files.append(os.path.join(tasks_dir, name))
  elif arg:
    candidate = os.path.join(tasks_dir, f"{arg}.yml")
    if os.path.exists(candidate):
      task_files.append(candidate)
    else:
      print(f"❌ Task '{arg}' not found in tasks/")
      return
  else:
    print("❌ Missing argument. Usage: python scan_channel.py ALL | task_name")
    return

  for tf in task_files:
    task = load_task_file(tf)
    if not task:
      print(f"❌ Skipping invalid task file: {tf}")
      continue
    print(f"\n=== Processing task: {os.path.basename(tf)} ===")
    process_task(task, os.path.basename(tf))

if __name__ == "__main__":
  main()
