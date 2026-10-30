# Drifting Gabors

Code accompanying the single-drift and double-drift paper by Sharif Saleki, Patrick Cavanagh and Peter Tse.

## Version and provenance

This is a source-only snapshot for manuscript **v24**, including **Figure 4 revision 4 (29 October 2026)**. Paper versions do not imply that every source file was rewritten: v24 inherits the v16 analysis rules, Figure 1 v11, and the v21 statistics, Figures 2–3 and movie scripts. `SOURCE_MANIFEST.json` records each archived source and its checksum.

The archived experiment is software **version 9**, not a new “v24 experiment”. It is retained as historical source, not certified as the exact collection-time executable or a ready-to-run replication.

No participant trial files, identifying records, gaze recordings, notebook outputs, manuscript documents, correspondence or calibration files are included. Supply the study data separately to run the analyses. Literature tables are included with source annotations.

## Analysis and figures

Install the analysis dependencies in a separate environment:

```sh
python -m pip install -r requirements-analysis.txt
```

Place the two preprocessed study tables at `data/preproc/speed-2.5_df-ori.csv` and `data/preproc/speed-2.5_df-len.csv`. From the repository root:

```sh
python analysis/figure1_v11.py output/figure1
python analysis/figures_v21.py data/preproc data/literature/figure4_points_v21.csv output/v21
python analysis/figure4_v24_r4.py output/v21/cell_means_v21.csv data/literature/figure4_points_v24.csv output/v24
```

`figures_v21.py` produces the current Figures 2–3 and saturation statistics, plus the superseded v21 Figure 4. Use **`output/v24/Figure4_r4.png` / `.pdf`** for the latest Figure 4, not the v21 comparison plot.

The inherited `analysis/DeValois_v16_analysis.ipynb` rebuilds preprocessed tables from the raw trial files and computes participant-level repeated-measures ANOVAs and paired tests. It remains explicitly labelled v16, with its original figure numbering; it is not a newly executed v24 notebook. Outputs were cleared for this public source snapshot. To use it, provide the twelve original trial CSVs in `data/raw/`, then:

```sh
cd analysis
jupyter nbconvert --to notebook --execute DeValois_v16_analysis.ipynb --output DeValois_v16_analysis_executed.ipynb --output-dir ../output
```

Every orientation setting counts as a report. Only reports more than 3 SD from their participant-by-condition mean are excluded; zero-variance cells are retained. Double-drift offsets use matched length multiplied by the sine of the illusory angle. The statistics reported in the paper are participant-level tests. The v21 script also retains the previously requested descriptive exponential saturation fits; these do not replace those tests.

## Demonstration movies

```sh
python analysis/make_movies_v21_r2.py output/movies
```

This requires FFmpeg and the DejaVu Sans Bold font. These are demonstrations, not exact experimental displays: revision 2 deliberately uses 0.55 cycles/degree and 4 Hz, whereas the paper reports about 0.44 cycles/degree and 5 Hz. Full movie rendering has not been rerun for this source snapshot.

## Archived experiment

`experiment/` contains the original PsychoPy source and configuration from the supplied Submission archive. It requires PsychoPy, NumPy, pandas, SciPy, matplotlib, seaborn and SR Research pylink. The optional contrast-measurement utility also imports `fips`, which was not supplied in the archive. The original imports expect `experiment/src` on Python’s module search path. For example, in a compatible Unix-like PsychoPy environment:

```sh
cd experiment
PYTHONPATH=src python run.py
```

This is an archival launch command, not a verified installation recipe. `run.py` defaults to a dummy tracker, but still imports pylink. Real EyeLink sessions need the tracker and display setup and calibration. Bundled sound assets and monitor geometry are retained; personal user entries were removed from `settings.json`. Do not reuse monitor geometry without measuring your display. Historical calibration arrays and participant/debrief files were excluded.

### Known collection-code discrepancy

The archived version-9 `double_drift.py` converts `GABOR_CYCLES = 0.8` as cycles/degree, while its commented earlier expression divides by the 1.8-degree patch size, giving about 0.44 cycles/degree, the collection value confirmed by the author and reported in the manuscript. The archive is left unchanged rather than silently “fixing” the historical program. This difference must be resolved before collecting replication data. Timing and hardware behavior have not been verified here; the original dropped-frame check was disabled.

## Licensing

See `THIRD_PARTY.md`. Existing third-party notices are retained. This snapshot does not assign a new license to the authors’ own code.
