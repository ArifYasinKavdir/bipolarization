import sys, json, time, subprocess
import numpy as np
from faster_whisper import WhisperModel
def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
for path in sys.argv[1:]:
    t0 = time.time()
    segs, info = model.transcribe(load(path), word_timestamps=True, beam_size=5, language="en", vad_filter=False)
    words = []
    for s in segs:
        for w in s.words:
            words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 3)})
    out = path.rsplit(".", 1)[0] + ".words.json"
    json.dump(words, open(out, "w"), indent=0)
    print(path, len(words), "words", round(time.time() - t0, 1), "s")
    print(" ".join(w["w"] for w in words))
