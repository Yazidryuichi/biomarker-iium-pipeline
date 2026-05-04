# Data Directory

**This directory is intentionally empty in the repository.**

EEG and behavioral data files contain identifiable participant information (children's names) and must NOT be committed to Git.

## How to set up data

1. Get access to the shared drive (contact Dandy or Yazid)
2. Copy the following files into this directory:

```
data/
  EDF/
    D0000795/
      X_M_X_Name_IGS_Eyes_Open.edf
      X_M_X_Name_IGS_Eyes_Closed.edf
      X_M_X_Name_IGS_1_Happy.edf
      X_M_X_Name_IGS_2_Calm.edf
      X_M_X_Name_IGS_3_Sad.edf
      X_M_X_Name_IGS_4_Scare.edf
    D0000796/
      ...
    (28 subject folders total)

  Behavioral/
    AUFEI-O_Cleaned.xlsx      (28 rows, AUFEI behavioral scores)
    Flanker_Test_Pilot.xlsx   (28 rows, Flanker task results)
    Digit_Span.xlsx           (28 rows, forward + backward digit span)
```

3. Verify by running:
   ```bash
   python pipeline.py --subject D0000795
   ```

## Data description

| File | Content | N |
|------|---------|---|
| EDF/ | 15-channel EEG at 250 Hz, EDF format | 28 subjects x 2-6 conditions (primary EO + EC; emotional opt-in via `--include-emotional`) |
| Behavioral/AUFEI-O_Cleaned.xlsx | Executive function questionnaire scores (WM, IC, CF, P, SF domains, Global EF) | 28 |
| Behavioral/Flanker_Test_Pilot.xlsx | Fish Flanker task (RT, accuracy, flanker effect, EZ-DDM parameters) | 28 |
| Behavioral/Digit_Span.xlsx | Forward + backward digit span scores | 28 |
