# CSE 547 Final Project

This repository contains the completed report package for the CSE 547 final project.

## Current Status

The full Report 2 pipeline has already been run on the desktop CUDA environment. The final artifacts are ready for review, writing polish, and submission.

Final selected models:

| Sensor | Final model | Validation accuracy | Validation weighted F1 |
|---|---:|---:|---:|
| RGB | RGB VGG16 refined | 0.9498 | 0.9480 |
| IR | IR CNN Arch D + Aug L2 | 0.9265 | 0.9227 |

Key completed outputs:

- `CSE547_FinalProject_Report2_Fuentes.ipynb` - final self-contained source notebook.
- `CSE547_FinalProject_Report2_Fuentes.pdf` - generated 5-page Report 2 PDF.
- `REPORT2_VIDEO_OUTLINE.md` - concise outline for the required video walkthrough.
- `figures/` - all plots, result JSON files, prediction CSVs, and error-analysis figures used by the report.
- `report2_pipeline.py` - script version of the final notebook pipeline.

## What To Submit

Submit these files to Blackboard unless your instructor requests additional files:

1. `CSE547_FinalProject_Report2_Fuentes.pdf`
2. `CSE547_FinalProject_Report2_Fuentes.ipynb`
3. Link to the recorded video walkthrough

Use `REPORT2_VIDEO_OUTLINE.md` as the speaking outline for the video.

## How To Finish Report Writing On A Laptop

The report data, plots, and final model selections are already generated. On the laptop, focus on editing/polishing the written report rather than retraining.

Recommended workflow:

1. Open `CSE547_FinalProject_Report2_Fuentes.pdf`.
2. Review the 5-page layout against the assignment prompt.
3. Use the figures in `figures/` if you need to rebuild or improve any page manually.
4. Use `REPORT2_VIDEO_OUTLINE.md` to record the 10-minute video.
5. Submit the PDF, notebook, and video link.

## Reproducibility Notes

The notebook is designed to reuse cached result files when present. The training run has already produced the required figures and JSON summaries.

A complete from-scratch rerun requires the original local dataset folders and manifests, which are intentionally not committed because they are large and contain machine-specific absolute paths:

- `data/`
- `train_rgb_patches/`
- `train_thermal_patches/`
- `rgb_full_frames/`
- `manifests/`
- `checkpoints/`

To rerun training on another machine, copy the dataset/checkpoint folders separately or regenerate the manifests after placing the dataset on that machine.

## Environment Used

The successful run used CUDA with PyTorch and torchvision. The Python dependencies are listed in `requirements.txt`.

For laptop review without retraining, opening the PDF and notebook does not require the full CUDA environment.
