"""
Digit Span Audio Generator (Edge TTS - Natural Voice)
=====================================================
Protocol: Lampiran F - Dokumen Pengambilan Data v2

Install & Run:
    pip install edge-tts imageio-ffmpeg
    python generate_digit_span.py

Output: ./digit_span_audio/ (30 MP3 files: 4 practice + 26 main)
"""

import edge_tts
import asyncio
import os
import wave
import struct
import subprocess
import shutil
import tempfile

# ============================================================
# FIND FFMPEG
# ============================================================
def get_ffmpeg():
    if shutil.which("ffmpeg"): return "ffmpeg"
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(p): return p
    except ImportError: pass
    print("ERROR: pip install imageio-ffmpeg")
    raise SystemExit(1)

FFMPEG = get_ffmpeg()

# ============================================================
# SEQUENCES
# ============================================================

# Practice trials (distinct from main trials)
PRACTICE_FW = [[6, 4], [8, 2]]
PRACTICE_BW = [[4, 8], [3, 1]]

# Main trials from protocol Lampiran F
FORWARD = [
    [3,7],[5,1],[4,9,2],[6,1,8],[3,8,5,2],[7,1,4,9],
    [2,9,6,1,3],[5,3,8,4,7],[1,7,3,9,5,2],[8,4,6,1,9,3],
    [3,5,1,8,6,2,9],[7,2,4,9,1,5,3],
    [5,9,3,7,1,6,2,8],[4,1,8,3,6,9,5,2],
]
BACKWARD = [
    [2,5],[6,3],[5,7,4],[2,9,1],[7,3,8,1],[4,6,2,9],
    [1,8,5,3,6],[9,2,7,4,1],[3,6,9,1,7,4],[5,8,2,4,1,7],
    [8,1,5,9,3,6,2],[4,7,3,1,8,5,9],
]

DIGIT_WORDS = {
    0:"nol",1:"satu",2:"dua",3:"tiga",4:"empat",
    5:"lima",6:"enam",7:"tujuh",8:"delapan",9:"sembilan"
}

# ============================================================
# SETTINGS
# ============================================================
VOICE = "id-ID-ArdiNeural"
# VOICE = "id-ID-GadisNeural"
RATE = "-10%"
ONSET_INTERVAL_MS = 1200
LEAD_IN_MS = 800
TAIL_MS = 2000
FADE_SAMPLES = 200
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2

OUTPUT_DIR = "./digit_span_audio"

# ============================================================
# AUDIO HELPERS
# ============================================================
digit_cache = {}

def make_silence_pcm(duration_ms):
    n = int(SAMPLE_RATE * duration_ms / 1000)
    return b'\x00' * (n * SAMPLE_WIDTH * CHANNELS)

def apply_fade(pcm, fade=FADE_SAMPLES):
    samples = list(struct.unpack(f"<{len(pcm)//2}h", pcm))
    n = len(samples); f = min(fade, n//2)
    for i in range(f):
        r = i/f
        samples[i] = int(samples[i]*r)
        samples[n-1-i] = int(samples[n-1-i]*r)
    return struct.pack(f"<{n}h", *samples)

def trim_silence_pcm(pcm, threshold=200, chunk=100):
    samples = struct.unpack(f"<{len(pcm)//2}h", pcm)
    n = len(samples); start = 0; end = n
    for i in range(0, n, chunk):
        if max(abs(s) for s in samples[i:i+chunk]) > threshold:
            start = max(0, i-chunk); break
    for i in range(n, 0, -chunk):
        if max(abs(s) for s in samples[max(0,i-chunk):i]) > threshold:
            end = min(n, i+chunk); break
    t = samples[start:end]
    return struct.pack(f"<{len(t)}h", *t)

def normalize_pcm(pcm, target_dbfs=-18):
    samples = struct.unpack(f"<{len(pcm)//2}h", pcm)
    if not samples: return pcm
    peak = max(abs(s) for s in samples) or 1
    target_peak = int(32767 * (10 ** (target_dbfs / 20)))
    factor = target_peak / peak
    norm = [max(-32768, min(32767, int(s*factor))) for s in samples]
    return struct.pack(f"<{len(norm)}h", *norm)

def pcm_duration_ms(pcm):
    return (len(pcm) // (SAMPLE_WIDTH * CHANNELS)) / SAMPLE_RATE * 1000

async def get_digit_pcm(digit):
    if digit in digit_cache: return digit_cache[digit]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3 = f.name
    try:
        await edge_tts.Communicate(DIGIT_WORDS[digit], VOICE, rate=RATE).save(mp3)
        result = subprocess.run([
            FFMPEG, "-y", "-i", mp3, "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS), "-f", "s16le", "-acodec", "pcm_s16le", "-"
        ], capture_output=True, check=True)
        pcm = result.stdout
    finally:
        os.unlink(mp3)
    pcm = trim_silence_pcm(pcm)
    pcm = apply_fade(pcm)
    digit_cache[digit] = pcm
    return pcm

async def build_sequence(digits, output_path):
    combined = bytearray()
    combined.extend(make_silence_pcm(LEAD_IN_MS))
    for i, d in enumerate(digits):
        dpcm = await get_digit_pcm(d)
        combined.extend(dpcm)
        if i < len(digits)-1:
            gap = max(80, ONSET_INTERVAL_MS - pcm_duration_ms(dpcm))
            combined.extend(make_silence_pcm(gap))
    combined.extend(make_silence_pcm(TAIL_MS))
    combined = bytearray(normalize_pcm(bytes(combined)))

    wav_path = output_path.replace(".mp3", ".wav")
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE); wf.writeframes(bytes(combined))
    subprocess.run([
        FFMPEG, "-y", "-i", wav_path, "-codec:a", "libmp3lame",
        "-b:a", "192k", "-ar", "44100", output_path
    ], capture_output=True, check=True)
    os.remove(wav_path)
    return pcm_duration_ms(bytes(combined))

# ============================================================
# MAIN
# ============================================================
async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("DIGIT SPAN AUDIO GENERATOR")
    print(f"Voice: {VOICE} | Rate: {RATE} | IOI: {ONSET_INTERVAL_MS}ms")
    print(f"FFmpeg: {FFMPEG}")
    print("=" * 60)

    all_files = []
    sets = [
        ("PRACTICE FORWARD", PRACTICE_FW, "PFW"),
        ("PRACTICE BACKWARD", PRACTICE_BW, "PBW"),
        ("FORWARD", FORWARD, "FW"),
        ("BACKWARD", BACKWARD, "BW"),
    ]

    for label, sequences, prefix in sets:
        print(f"\n--- {label} ---")
        for i, seq in enumerate(sequences):
            if prefix.startswith("P"):
                fname = f"{prefix}_T{i+1}_span{len(seq)}.mp3"
                tag = f"  Practice {i+1}"
            else:
                level = (i//2)+1; trial = (i%2)+1
                fname = f"{prefix}_L{level}_T{trial}_span{len(seq)}.mp3"
                tag = f"  L{level} T{trial}"

            fpath = os.path.join(OUTPUT_DIR, fname)
            dur = await build_sequence(seq, fpath)
            seq_str = "-".join(map(str, seq))
            words = " ".join(DIGIT_WORDS[d] for d in seq)
            print(f"{tag:12s} | span {len(seq)} | {seq_str:25s} | {words:30s} | {dur/1000:.1f}s ✓")
            all_files.append(fpath)

    print(f"\n{'='*60}")
    print(f"✓ Generated {len(all_files)} files in {OUTPUT_DIR}/")

if __name__ == "__main__":
    asyncio.run(main())