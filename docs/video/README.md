# Narrated explainer video

Source for the 2:40 narrated video that walks through the bipolarization method. It reuses the drawing and scoring code of the interactive explainer in `docs/explainer/src/`. The rendered MP4 (1920×1080, 30 fps, about 16 MB) is not stored in git.

| File | What it is |
|---|---|
| `narration.txt` | Voice-over script. `[pause]` tags are ElevenLabs audio tags. |
| `timing.json` | Start and end time of every script word in the narration take that was used, from `tools/align.py`. |
| `video.js`, `video.css` | The composition. Every frame is a pure function of time `T`, and animations are keyed to spoken cue phrases via `cue("...")`. |
| `tools/` | Transcription, alignment, page build and frame rendering scripts. |

## Narration

The voice-over was generated with ElevenLabs (`eleven_v4`, voice "Aabo – Engaging Confident Explainer Voice", `KeU8nqWFDbaoi0QVUjD3`), using take `UCqGL0JA8ewuzkl86n6K` (155.9 s). The audio file is not committed. To re-render without generating a new take, extract the narration from the published MP4 and remove the 0.8 s lead-in:

```bash
ffmpeg -i bipolarization-explainer.mp4 -ss 0.8 -t 155.873 -vn narration.wav
```

With a new take, regenerate `timing.json` and update `AUDIO_DUR` at the top of `video.js`:

```bash
python tools/transcribe.py take.mp3                      # needs faster-whisper; writes take.words.json
python tools/align.py narration.txt take.words.json timing.json
```

## Render

Needs Python Playwright with Chromium, and ffmpeg with libx264.

```bash
python docs/video/tools/build_video.py docs/video docs/explainer/src /tmp/video/video.html
python docs/video/tools/render.py /tmp/video/video.html /tmp/video 3     # 3 parallel workers → seg_00.mp4 …
cd /tmp/video && printf "file 'seg_00.mp4'\nfile 'seg_01.mp4'\nfile 'seg_02.mp4'\n" > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy video_noaudio.mp4
ffmpeg -i video_noaudio.mp4 -i narration.wav \
  -filter_complex "[1:a]adelay=800:all=1,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,apad=whole_dur=160.073[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -t 160.073 -movflags +faststart bipolarization-explainer.mp4
```

The AKP × CHP cell counts shown in the video are simulated to reproduce the paper's published Figure 2(c). The stated results (S_pol = 0.453, the per-person scores and their intervals, S_con = −0.199, the bootstrap interval) are the paper's published values.
