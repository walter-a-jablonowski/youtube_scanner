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
from google import genai

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
  client = genai.Client(api_key=LLM_KEY)
else:
  raise SystemExit(f"❌ Unsupported LLM provider: {LLM_PROVIDER}")

# === FUNCTIONS ===
def fetch_video_ids_via_ytdlp(limit=100, channel_url=None, channel_label=None):
  """Use yt-dlp to extract videos with minimal metadata. Requires yt-dlp.

  Returns a list of dicts: {id, upload_date, timestamp, channel, title}
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
          title = e.get('title') or ''
          item = {
            'id': vid,
            'upload_date': upload_date,
            'timestamp': ts,
            'channel': channel_name,
            'title': title,
          }
          # Enrich missing fields with a per-video metadata call
          if not (item['timestamp'] or item['upload_date']) or not item['channel'] or not item['title']:
            try:
              vinfo = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
              item['upload_date'] = item['upload_date'] or vinfo.get('upload_date')
              item['timestamp']   = item['timestamp'] or vinfo.get('timestamp') or vinfo.get('release_timestamp')
              item['channel']     = item['channel'] or vinfo.get('channel') or vinfo.get('uploader')
              item['title']       = item['title'] or vinfo.get('title') or ''
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

def generate_entry_id(upload_date, title, video_id):
  """Generate ENTRY_ID: YY-MM-DD__TITLE__VID_ID
  - YY-MM-DD: 2-digit year format
  - TITLE: first 11 chars after replacing non-alphanumeric with underscore
  - VID_ID: youtube video id
  """
  # Format date as YY-MM-DD
  if upload_date and len(upload_date) == 8:  # YYYYMMDD
    date_part = f"{upload_date[2:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
  else:
    date_part = "00-00-00"
  
  # Clean title: replace non-alphanumeric with underscore, take first 11 chars
  clean_title = re.sub(r'[^a-zA-Z0-9]', '_', title or 'untitled')
  title_part = clean_title[:11]
  
  return f"{date_part}__{title_part}__{video_id}"

def execute_prompt_task(prompt_template, variables):
  """Execute a prompt task using the LLM with variable substitution."""
  prompt = prompt_template
  for key, value in variables.items():
    placeholder = "{" + key + "}"
    prompt = prompt.replace(placeholder, str(value))
  
  try:
    response = client.models.generate_content(
      model=LLM_MODEL,
      contents=prompt
    )
    return response.text.strip()
  except Exception as e:
    print(f"⚠️ LLM error: {e}")
    return None

def extract_plain_text(transcript):
  """Extract plain text from transcript by removing timestamps and formatting tags."""
  if not transcript:
    return ""
  
  lines = transcript.split('\n')
  plain_lines = []
  
  for line in lines:
    line = line.strip()
    if not line:
      continue
    # Remove common transcript formatting patterns
    # Remove timestamps like [00:00:00] or 00:00:00 or <00:00:00>
    line = re.sub(r'[\[<]?\d{1,2}:\d{2}(?::\d{2})?[\]>]?', '', line)
    # Remove speaker tags like "Speaker:" or "[Speaker]"
    line = re.sub(r'^[\[<]?[A-Za-z\s]+[\]>]?:', '', line)
    # Remove XML-like tags
    line = re.sub(r'<[^>]+>', '', line)
    line = line.strip()
    if line:
      plain_lines.append(line)
  
  return ' '.join(plain_lines)

def write_output_file(file_path, content, action):
  """Write content to file with specified action (overwrite or append)."""
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  
  if action == "append":
    with open(file_path, "a", encoding="utf-8") as f:
      f.write(content + "\n")
  else:  # overwrite
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(content)

def substitute_variables(template, variables):
  """Replace {VAR} placeholders in template with values from variables dict."""
  result = template
  for key, value in variables.items():
    placeholder = "{" + key + "}"
    result = result.replace(placeholder, str(value))
  return result

# === MAIN ===
def process_task(task_config, task_name):
  """Process a task with new multi-output structure."""
  # Extract task settings
  base_folder = task_config.get("baseFolder", "output")
  tasks = task_config.get("tasks", [])
  max_videos = int(task_config.get("max_videos", 100))
  skip_processed = bool(task_config.get("skip_processed", True))
  polite_delay = int(task_config.get("polite_delay_sec", 1))
    
  # Create -this.yml file in output folder if it doesn't exist
  task_output_folder = os.path.join(base_folder, task_name.replace('.yml', ''))
  this_yml_path = os.path.join(task_output_folder, '-this.yml')
  version = task_config.get("version", "unknown")
  
  if not os.path.exists(this_yml_path):
    try:
      os.makedirs(task_output_folder, exist_ok=True)
      with open(this_yml_path, 'w', encoding='utf-8') as f:
        yaml.dump({'version': version}, f, default_flow_style=False, allow_unicode=True)
      print(f"📝 Created task config: {this_yml_path} (version: {version})")
    except Exception as e:
      if DEBUG_MODE:
        print(f"[debug] Failed to create -this.yml: {e}")
  
  # Build channels from task
  channels = []  # list of (label, url)
  for label, data in (task_config.get("channels") or {}).items():
    url = (data or {}).get("url")
    if url:
      channels.append((label, url))
  
  if not channels:
    print(f"❌ Task '{task_name}' has no channels configured.")
    return
  
  if not tasks:
    print(f"❌ Task '{task_name}' has no tasks configured.")
    return
  
  # Discover videos across channels
  all_items = []
  if USE_YTDLP:
    if yt_dlp is None and DEBUG_MODE:
      print("[debug] yt-dlp is not installed; cannot use yt-dlp discovery")
    else:
      for label, url in channels:
        if DEBUG_MODE:
          print(f"[debug] Discovering channel {label}: {url}")
        items = fetch_video_ids_via_ytdlp(max_videos, channel_url=url, channel_label=label)
        if DEBUG_MODE:
          print(f"[debug] {label}: collected {len(items)} IDs")
        all_items.extend(items)
  
  if not all_items:
    print("🔎 Found 0 videos via yt-dlp. Enable run.debug for details.")
    print(f"❌ No videos to process for task '{task_name}'.")
    return
  
  # Setup log.csv path
  log_csv_path = os.path.join(base_folder, task_name.replace('.yml', ''), 'log.csv')
  os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
  
  # Read existing processed URLs from log.csv
  processed_urls = set()
  if skip_processed and os.path.exists(log_csv_path):
    try:
      with open(log_csv_path, "r", newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf, delimiter=';')
        next(reader, None)  # Skip header
        for row in reader:
          if len(row) >= 5:
            processed_urls.add(row[4])  # Video URL column
    except Exception:
      processed_urls = set()
  
  # Initialize log.csv if needed
  log_exists = os.path.exists(log_csv_path)
  log_empty = False
  if log_exists:
    try:
      log_empty = os.path.getsize(log_csv_path) == 0
    except Exception:
      log_empty = False
  
  with open(log_csv_path, "a", newline="", encoding="utf-8") as log_f:
    log_writer = csv.writer(log_f, delimiter=';')
    if not log_exists or log_empty:
      log_writer.writerow(["Channel", "Date", "Title", "State", "Video URL", "Folder"])
    
    total = len(all_items)
    for i, item in enumerate(all_items, 1):
      vid = item['id']
      url = f"https://www.youtube.com/watch?v={vid}"
      
      if url in processed_urls:
        print(f"\n[{i}/{total}] Skipping already processed {url}")
        continue
      
      print(f"\n[{i}/{total}] Processing {url}")
      
      # Generate ENTRY_ID
      upload_date = item.get('upload_date', '')
      title = item.get('title', '')
      entry_id = generate_entry_id(upload_date, title, vid)
      channel_name = item.get('channel') or ''
      
      # Format date for log (YY-MM-DD)
      if upload_date and len(upload_date) == 8:
        date_str = f"{upload_date[2:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
      else:
        date_str = ""
      
      # Fetch transcript
      transcript = get_transcript(vid)
      if not transcript:
        print("  ❌ No transcript found, skipping.")
        time.sleep(polite_delay)
        continue
      
      # Execute tasks in sequence
      task_results = {}  # Store results by task name
      
      for task_def in tasks:
        task_type = task_def.get("type")
        task_task_name = task_def.get("name", "")
        output_file = task_def.get("output_file", "")
        action = task_def.get("action", "overwrite")
        
        # Build variables for substitution
        variables = {
          "CHANNEL_NAME": channel_name,
          "ENTRY_ID": entry_id,
          "transcript": transcript,
        }
        # Add previous task results
        variables.update(task_results)
        
        # Substitute variables in output_file path
        full_path = os.path.join(base_folder, task_name.replace('.yml', ''), substitute_variables(output_file, variables))
        
        if task_type == "save_transcript":
          # Save transcript to file
          write_output_file(full_path, transcript, action)
          task_results[task_task_name] = transcript
          print(f"  💾 Saved transcript to {full_path}")
        
        elif task_type == "save_transcript_plain":
          # Extract plain text from input and save
          input_name = task_def.get("input", "transcript")
          input_text = variables.get(input_name, "")
          plain_text = extract_plain_text(input_text)
          write_output_file(full_path, plain_text, action)
          task_results[task_task_name] = plain_text
          print(f"  💾 Saved plain text to {full_path}")
        
        elif task_type == "prompt":
          # Execute prompt with LLM
          prompt_template = task_def.get("prompt", "")
          result = execute_prompt_task(prompt_template, variables)
          
          if result:
            write_output_file(full_path, result, action)
            task_results[task_task_name] = result
            print(f"  ✅ {task_task_name}: {result[:80]}..." if len(result) > 80 else f"  ✅ {task_task_name}: {result}")
          else:
            print(f"  ⚠️ {task_task_name}: Failed to generate result")
            task_results[task_task_name] = "n/a"
      
      # Write to log.csv
      folder_path = f"{channel_name}/{entry_id}"
      log_writer.writerow([channel_name, date_str, title, "", url, folder_path])
      
      time.sleep(polite_delay)
  
  print(f"\n✅ Done! Log saved to {log_csv_path}")

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
      if name.lower().endswith('.yml') and not name.startswith('.') and not name.startswith('_'):
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
