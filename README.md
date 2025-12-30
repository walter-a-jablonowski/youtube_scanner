# YouTube Scanner

A script to scan YouTube channels for videos, fetch transcripts, and process them with multiple tasks (save transcripts, generate summaries, etc.) using an LLM.

- Video discovery and processing

  - Discovers video IDs via yt-dlp from channel URLs (no YouTube Data API)
  - Retrieves captions via `youtube_transcript_api`, with yt-dlp `.vtt` subtitle fallback
  - Processes videos through multiple tasks (save transcript, LLM prompts, etc.)
  - Creates organized output folders per video with unique ENTRY_ID
  - If a transcript is unavailable (private/restricted/no captions), the video is skipped

- Logic

  - Each task creates a `log.csv` file tracking processed videos
  - Subsequent runs skip URLs already present in the log
  - Multiple output files per video based on task configuration

## Installation

- **Python** 3.9+
  ```bash
  pip install -r requirements.txt
  # or
  pip install google-generativeai youtube-transcript-api requests pyyaml
  python -m pip install yt-dlp
  ```
- **Update config.yml**
- **API key**
  - set in api_keys.yml (gitignored)
    ```yml
    llm: "MY_API_KEY"
    ```
  - or env (priority)
    ```bash

    set YT_SCANNER_API_KEY=your_api_key_here
    # powershell: $env:YT_SCANNER_API_KEY="your_api_key_here"

    # or permanent

    [System.Environment]::SetEnvironmentVariable("YT_SCANNER_API_KEY", "your_api_key_here", "User")
    # powershell: [System.Environment]::SetEnvironmentVariable("YT_SCANNER_API_KEY", "your_api_key_here", "Machine")
    # restert terminal  echo $env:YT_SCANNER_API_KEY  echo %YT_SCANNER_API_KEY%
    ```
- **Define task:** see `tasks/_sample.yml` for the new multi-output structure


## Run

```
python scan_channels.py ALL        # files with "." or "_" in front are excluded
python scan_channels.py task_name
```

Warnings can be ignored for transcript-only usage (are from yt-dlp)


### log.csv format

```
Channel;Date;Title;State;Video URL;Folder
Channel Name;25-10-16;Video Title;;https://www.youtube.com/watch?v=...;Channel Name/25-10-16__Video_Title__VIDEO_ID
```

- **ENTRY_ID format**: `YY-MM-DD__TITLE__VID_ID`
  - `YY-MM-DD`: Upload date (2-digit year)
  - `TITLE`: First 11 chars of video title (non-alphanumeric replaced with `_`)
  - `VID_ID`: YouTube video ID

### Task Configuration

see `tasks/_sample.yml`

Sample output structure

Each task creates the following structure:

```
output/
  TASK_NAME/
    log.csv                                    # tracking file
    CHANNEL_NAME/
      ENTRY_ID/                                # YY-MM-DD__TITLE__VID_ID
        original.txt                           # transcript
    summary.md                                 # appended summaries
```

## License

MIT, please respect youtube terms of use
