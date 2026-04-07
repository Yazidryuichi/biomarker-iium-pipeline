# Data Directory

**This directory is intentionally empty in the repository.**

EEG and behavioral data files contain identifiable participant information (children's names) and must NOT be committed to Git.

## How to set up data

1. Get access to the shared drive (contact Dandy or Yazid)
2. Copy the following files into this directory:

```
data/
  EDF_Files/
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

  AUFEI-O_Cleaned.xlsx      (28 rows, AUFEI behavioral scores)
  Flanker_Test_Pilot.xlsx   (28 rows, Flanker task results)
  Digit_Span.xlsx           (28 rows, forward + backward digit span)
```

3. Verify by running:
   ```bash
   python run_all.py --subject D0000795
   ```

## Data description

| File | Content | N |
|------|---------|---|
| EDF_Files/ | 15-channel EEG at 250 Hz, EDF format | 28 subjects x 6 conditions |
| AUFEI-O_Cleaned.xlsx | Executive function questionnaire scores (WM, IC, CF, P, SF domains) | 28 |
| Flanker_Test_Pilot.xlsx | Fish Flanker task (RT, accuracy, flanker effect, DDM parameters) | 28 |
| Digit_Span.xlsx | Forward + backward digit span scores | 28 |
