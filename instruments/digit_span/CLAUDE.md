# Digit Span Test (digit_span)

## What this project is

A browser-based **Digit Span** working-memory test (forward and backward) with Indonesian-language audio stimuli, plus the Python script that generates those audio files. An administrator runs the test in a web browser; the participant listens to spoken digit sequences and the admin records responses, which export to Excel.

Protocol source: "Lampiran F – Dokumen Pengambilan Data v2".

## Files

- `digit_span_admin.html` — main administration tool. Plays sequences, records correct/incorrect per trial, exports results to XLSX (via the SheetJS CDN). Auto-advances / can stop on the standard 2-consecutive-failures rule.
- `digit_span_admin_1by1.html` — alternative administration UI that presents digits one-by-one.
- `generate_digit_span.py` — regenerates the audio stimuli using Edge TTS (`id-ID-ArdiNeural` voice) + FFmpeg.
- `digit_span_audio/` — 30 generated MP3 stimuli (4 practice + 26 main trials).
- `output/` — collected results exported from the admin tool: `Day1..Day4_DigitSpan_<date>_n<N>.xlsx` (pilot, March 2026) plus `FULL.xlsx` (merged). Moved here from `PILOT/digit_span/` on 2026-07-27 so the instrument owns its own output.
- `saved_page/` — **gitignored, contains participant data.** A browser "save page as" taken partway through a session on 2026-03-09: the rendered results table still holds seven children's given names (`Prabu`, `Ukasya`, `Aqueen`, `Aqilah`, `Keana`, `Shafia`, `Hafiz`). It is a session artefact, not an instrument file. Do not commit it, and do not treat it as a blank offline copy of the tool — `digit_span_admin.html` is the blank copy. Its one genuinely reusable part is the vendored `xlsx.full.min.js`, which would let the tool export without a CDN; if offline export is ever needed, vendor that script deliberately rather than resurrecting this page.

## Important structural constraint

The HTML tools load audio by the **relative path** `digit_span_audio/<FILE>.mp3` (see `digit_span_admin.html:196` and `:253`), and `generate_digit_span.py` writes to `./digit_span_audio/`. **Keep the HTML files and the `digit_span_audio/` folder as siblings** — do not move them into separate subfolders or the audio will 404. This is why the project is intentionally kept flat.

## Audio filename convention

`<PREFIX>_L<level>_T<trial>_span<N>.mp3` for main trials, `<PREFIX>_T<trial>_span<N>.mp3` for practice.

| Prefix | Meaning |
|---|---|
| FW | Forward (main) |
| BW | Backward (main) |
| PFW | Practice Forward |
| PBW | Practice Backward |

`span<N>` = number of digits in the sequence. Forward goes span 2→8, backward span 2→7, two trials (T1/T2) per level.

## Regenerating audio

```
pip install edge-tts imageio-ffmpeg
python generate_digit_span.py     # writes ./digit_span_audio/ (30 files)
```

The exact digit sequences are hardcoded in the `FORWARD` / `BACKWARD` / `PRACTICE_*` lists in the script. Timing: 1200 ms inter-onset interval, 800 ms lead-in, 2000 ms tail, normalized to −18 dBFS. Digits are spoken in Indonesian (nol, satu, dua, …).
