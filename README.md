# YouTube Scanner

A simple script to scan a YouTube channel for videos, fetch available transcripts, and extract a short summary per video using an LLM.

- Video discovery and summarization

  - Discovers video IDs via yt-dlp from `youtube.channel_url` (no YouTube Data API)
  - Retrieves captions via `youtube_transcript_api`, with yt-dlp `.vtt` subtitle fallback
  - Summarizes a short idea with your configured LLM model
  - If a transcript is unavailable (private/restricted/no captions), the script writes `n/a`

- Logic

  - The first run creates/appends to the CSV defined in `run.output_file`
  - Subsequent runs will skip URLs already present in the CSV

## Installation

- Python 3.9+
  ```bash
  pip install google-generativeai youtube-transcript-api requests pyyaml
  python -m pip install yt-dlp
  ```
- Update config.yml
- API key
  - set in api_keys.yml
  - or env (priority)
    ```bash

    set YT_SCANNER_API_KEY=your_api_key_here
    # powershell: $env:YT_SCANNER_API_KEY="your_api_key_here"

    # or permanent

    [System.Environment]::SetEnvironmentVariable("YT_SCANNER_API_KEY", "your_api_key_here", "User")
    # powershell: [System.Environment]::SetEnvironmentVariable("YT_SCANNER_API_KEY", "your_api_key_here", "Machine")
    # restert terminal  echo $env:YT_SCANNER_API_KEY  echo %YT_SCANNER_API_KEY%
    ```
- Define task: see tasks/_sample.yml


## Run

```
python scan_channels.py ALL
python scan_channels.py task_name
```

Warnings can be ignored for transcript-only usage (are from yt-dlp)


## License

MIT, please respect youtibe terms of use
