# Report 2 Video Outline

## First 5 minutes: notebook and code
1. Open `CSE547_FinalProject_Report2_Fuentes.ipynb`.
2. Show setup: seeds, manifests, device, checkpoint reuse, and class map.
3. Explain Part 3 VGG16: pretrained conv base, three freeze settings, two dense layers.
4. Explain Part 4 AE: six encoder configs, frozen features, two dense regularization heads.
5. Show generated artifacts in `figures/` and `checkpoints/`.

## Second 5 minutes: report and results
1. Open `CSE547_FinalProject_Report2_Fuentes.pdf`.
2. Compare Part 3 VGG16 results to RGB Part 1 Arch D.
3. Compare Part 4 AE results to IR Part 1 Arch D.
4. Discuss final model choices:
   - RGB: RGB VGG16 refined with validation F1 0.9480
   - IR: IR CNN Arch D + Aug L2 with validation F1 0.9227
5. Explain misclassified samples and paired RGB/IR disagreement examples.
6. Close with blind-test strategy: use validation-selected best models only, no withheld labels.
