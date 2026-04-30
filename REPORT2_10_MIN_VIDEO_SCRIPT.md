# Report 2 Video Script

Use this file as the speaking script for the required video recording.

Submit/show these files:
- Report PDF: `CSE547_FinalProject_Report2_Fuentes.pdf`
- Notebook to show and submit: `CSE547_FinalProject_Report2_Submission.ipynb`

Do not submit or show `CSE547_FinalProject_Report2_Fuentes.ipynb`; it is the older generated notebook that causes VS Code/Pylance display issues.

## Before Recording

1. Open VS Code in `C:\Users\Tonito\.vscode\Final-Exam-CSE-547`.
2. Open `CSE547_FinalProject_Report2_Submission.ipynb`.
3. Open `CSE547_FinalProject_Report2_Fuentes.pdf` in a PDF viewer.
4. Keep the `figures/` folder visible in the Explorer if possible.
5. Make sure the notebook kernel is `final-exam-cse547` or the project Python venv.

## 0:00-0:30 - Opening

Say:

"This is my CSE 547 final project Report 2 walkthrough. The project is object recognition for autonomous driving using RGB and infrared image patches from the FLIR ADAS v2 dataset. Report 1 covered the custom CNN baselines and regularization experiments. In this second report, I complete Parts 3 through 6: RGB transfer learning using VGG16, IR autoencoder features, RGB-vs-IR comparison and misclassification analysis, and final model refinement for blind-test performance."

Show:
- The repo folder in VS Code.
- The notebook `CSE547_FinalProject_Report2_Submission.ipynb`.

## 0:30-1:15 - Notebook Structure

Say:

"This is the notebook I am submitting. I used this cleaned submission notebook instead of the generated development notebook because it avoids static-analysis issues in VS Code while still embedding the full source code. The first cell sets runtime options. `REUSE_CHECKPOINTS` is true so the notebook reuses the trained checkpoints and cached result files. `SMOKE_TEST` is false, so it runs the full validation/report pipeline."

Show:
- Cell 1 with `REUSE_CHECKPOINTS = True`, `SMOKE_TEST = False`, `SKIP_TRAINING = False`.

## 1:15-2:15 - Embedded Source Code

Say:

"The second code cell embeds the full pipeline source code as a string and executes it. This includes the model definitions, data loading, evaluation, plotting, prediction mining, report generation, and output verification. The notebook is still self-contained because the source code is inside the notebook, but Pylance does not try to type-check every PyTorch line as an active notebook cell."

Show:
- The cell that defines `PIPELINE_SOURCE`.
- The line `exec(compile(...), pipeline)`.
- The output `Pipeline functions loaded: True`.

## 2:15-3:30 - Run the Pipeline

Say:

"The final cell calls `run_all`. This rebuilds the Report 2 artifacts from the trained checkpoints and cached JSON result files. It verifies that all required outputs are present. The notebook regenerates the figures, prediction CSVs, final model selection, PDF report, and video outline."

Action:
- If you are comfortable waiting about 45 seconds, click `Run All` at the beginning of the video.
- Otherwise, show the already-executed output.

Show:
- Output lines for Part 3, Part 4, Part 6, Part 5, and final PDF generation.
- The final output `All required outputs are present.`
- The final returned value `[]`, which means no missing outputs.

## 3:30-4:30 - Part 3 Code Explanation

Say:

"For Part 3, I used a pretrained VGG16 convolutional base for RGB only. I added two dense layers on top and tested three freezing strategies. F1 freezes all convolutional blocks, F2 unfreezes block 5, and F3 unfreezes blocks 4 and 5. I used differential learning rates so the pretrained layers train more conservatively than the new classifier head."

Show:
- The notebook output showing `Part 3: RGB VGG16 transfer learning`.
- The figure file `figures/part3_rgb_vgg16_freeze.png` if useful.

## 4:30-5:00 - Part 4 and Part 5 Code Explanation

Say:

"For Part 4, I trained six convolutional autoencoder configurations on IR patches, then froze each encoder and trained two dense classifier variants, giving 12 total options. For Part 5, the notebook generates predictions for the best RGB and IR models, mines misclassified examples by class, and finds paired scene examples where RGB and IR disagree."

Show:
- Output lines for Part 4 and Part 5.
- The files in `figures/` beginning with `part4` and `part5`.

## 5:00-5:45 - Switch to Report PDF

Say:

"Now I will walk through the report. The report is a single five-page PDF, which matches the requirement limit. It includes the required figures and result summaries, since the prompt says figures not included in the report will not be graded."

Show:
- Open `CSE547_FinalProject_Report2_Fuentes.pdf`.
- Page 1.

## 5:45-6:45 - Page 1: Objective, Baselines, and Part 3

Say:

"Page 1 gives the project objective and compares the Report 1 baselines to the new RGB transfer-learning models. The best Report 1 RGB model was the custom CNN Arch D with validation F1 of 0.9185. The VGG16 transfer-learning model improved on that. The best VGG16 freezing option was F3, which unfreezes blocks 4 and 5 and reaches validation F1 of 0.9466."

Show:
- Baseline table.
- VGG16 freeze table.
- VGG16 figure.

## 6:45-7:30 - Page 2: Part 4 IR Autoencoder

Say:

"Page 2 shows the IR autoencoder feature experiment. I tested six autoencoder configurations and two classifier regularization heads, so there are 12 total options. The best autoencoder-based classifier was AE6-R1 with validation F1 of 0.7820. This was significantly below the supervised IR CNN baseline from Report 1, so the autoencoder features were useful to analyze but were not selected as the final IR model."

Show:
- Autoencoder result figure.
- AE comparison text/table.

## 7:30-8:15 - Page 3: Part 6 Final Models

Say:

"Page 3 summarizes final model selection and refinement. For RGB, the final model is the refined VGG16 model with validation F1 of 0.9480. For IR, the final model remains the supervised CNN Arch D with augmentation level 2, with validation F1 of 0.9227. The autoencoder did not outperform the supervised CNN, so I selected the stronger supervised IR model for the final blind-test strategy."

Show:
- Final model selection table.
- Final RGB and final IR bullet points.

## 8:15-9:15 - Pages 4 and 5: Part 5 Misclassification and Sensor Analysis

Say:

"Pages 4 and 5 address the comparison and analysis requirements. Page 4 shows misclassified validation examples by class for each sensor. Page 5 shows paired RGB and IR scene-level disagreements. Because RGB and IR annotations do not reliably match at the exact object-instance level, I use paired video-frame scenes for qualitative comparison. The general pattern is that RGB benefits from color, texture, and object boundary detail, while IR can be more stable when visible detail is weaker. Small classes and low-resolution objects remain difficult for both modalities."

Show:
- Page 4 misclassification grids.
- Page 5 RGB-correct/IR-wrong and IR-correct/RGB-wrong examples.

## 9:15-9:50 - Final Conclusions

Say:

"The main conclusions are: VGG16 transfer learning substantially improved RGB performance over the custom CNN baseline; IR autoencoder features did not outperform the supervised CNN; sensor comparison shows different failure modes for RGB and IR; and the final selected models are RGB VGG16 refined and IR CNN Arch D with augmentation level 2. These final models were selected using validation weighted F1 only, without using withheld blind-test labels."

Show:
- Page 3 final model table again.

## 9:50-10:00 - Closing

Say:

"For submission, I am uploading the five-page PDF report, the clean submission notebook, and a link to this video recording with access enabled."

Show:
- The files:
  - `CSE547_FinalProject_Report2_Fuentes.pdf`
  - `CSE547_FinalProject_Report2_Submission.ipynb`

## Submit

Submit separately on Blackboard:
1. `CSE547_FinalProject_Report2_Fuentes.pdf`
2. `CSE547_FinalProject_Report2_Submission.ipynb`
3. Video recording link with access enabled

Do not upload a ZIP file.
