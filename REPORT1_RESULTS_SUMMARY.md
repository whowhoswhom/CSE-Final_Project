# Report 1 Results Summary

This file summarizes the completed Part 1 and Part 2 experiments for both modalities.

## Part 1: CNN Architecture Comparison

### Infrared (IR)

| Architecture | Params | Best Epoch | Train Acc | Val Acc | Train F1 | Val F1 |
|---|---:|---:|---:|---:|---:|---:|
| Arch A | 880 | 41 | 0.7663 | 0.7251 | 0.7168 | 0.6520 |
| Arch B | 14,600 | 62 | 0.8702 | 0.8070 | 0.8507 | 0.7563 |
| Arch C | 148,200 | 64 | 0.9552 | 0.8616 | 0.9535 | 0.8340 |
| Arch D | 2,459,848 | 9 | 0.9766 | 0.8880 | 0.9772 | 0.8792 |

Key observations:
- Validation performance improved monotonically as model capacity increased.
- Arch D achieved the strongest IR validation performance.
- Overfitting also increased with model size, but the performance gain from Arch C to Arch D was still worthwhile for IR.

Recommended Part 1 conclusion:
- The 10x parameter-growth rule was satisfied for all consecutive architectures.
- Increasing model capacity substantially improved IR classification performance, with Arch D performing best at Val F1 = 0.8792.

### RGB

| Architecture | Params | Best Epoch | Train Acc | Val Acc | Train F1 | Val F1 |
|---|---:|---:|---:|---:|---:|---:|
| Arch A | 880 | 24 | 0.6249 | 0.5988 | 0.5785 | 0.5384 |
| Arch B | 14,600 | 64 | 0.8393 | 0.7959 | 0.8230 | 0.7665 |
| Arch C | 148,200 | 70 | 0.9468 | 0.8909 | 0.9453 | 0.8857 |
| Arch D | 2,459,848 | 39 | 0.9999 | 0.9201 | 0.9999 | 0.9185 |

Key observations:
- Validation performance again improved with model size.
- Arch D was the strongest raw RGB model.
- RGB outperformed IR in the best Part 1 architecture comparison (RGB Arch D Val F1 0.9185 vs IR Arch D Val F1 0.8792).
- Arch C showed a better "modest overfitting" profile than Arch D and was therefore used as the Part 2 RGB base architecture.

Recommended Part 1 conclusion:
- RGB benefited strongly from increased CNN capacity, and Arch D was the best-performing architecture with Val F1 = 0.9185.
- Because Arch D showed a larger train-validation gap, Arch C was a more defensible choice for Part 2 regularization experiments.

## Part 2: Regularization and Augmentation

### Infrared (base architecture: Arch D)

#### 2A: L2 Regularization

| Setting | Val Acc | Val F1 |
|---|---:|---:|
| L2 = 1e-5 | 0.8984 | 0.8984 |
| L2 = 1e-4 | 0.8862 | 0.8903 |
| L2 = 1e-3 | 0.8724 | 0.8491 |
| L2 = 1e-2 | 0.9064 | 0.8973 |

Best L2 by Val F1: `1e-5`

#### 2B: Dropout

| Setting | Val Acc | Val F1 |
|---|---:|---:|
| Dropout = 0.1 | 0.8984 | 0.8838 |
| Dropout = 0.25 | 0.8924 | 0.8743 |
| Dropout = 0.4 | 0.8977 | 0.8809 |
| Dropout = 0.6 | 0.8746 | 0.8420 |

Best dropout by Val F1: `0.1`

#### 2C: Data Augmentation

| Setting | Val Acc | Val F1 |
|---|---:|---:|
| Aug L1 | 0.9085 | 0.8967 |
| Aug L2 | 0.9263 | 0.9227 |
| Aug L3 | 0.8908 | 0.8837 |
| Aug L4 | 0.8865 | 0.8872 |

Best augmentation by Val F1: `Aug L2`

#### 2D: Combined Settings

| Setting | Val Acc | Val F1 |
|---|---:|---:|
| L2 = 1e-5 + Dropout = 0.1 | 0.8786 | 0.8601 |
| L2 = 1e-5 + Aug L2 | 0.9129 | 0.9035 |
| Dropout = 0.1 + Aug L2 | 0.8988 | 0.8837 |
| L2 + Dropout + Aug (all best) | 0.9057 | 0.8981 |

Best combined setting by Val F1: `L2 = 1e-5 + Aug L2`

IR conclusions:
- The strongest single IR Part 2 setting was augmentation level 2 with Val F1 = 0.9227.
- This improved over the Part 1 Arch D baseline (0.8792) by +0.0435 F1.
- Combining the individually best settings did not outperform the best augmentation-only result.
- For IR, moderate augmentation was the most effective regularization strategy.

### RGB (base architecture: Arch C)

#### 2A: L2 Regularization

| Setting | Val Acc | Val F1 |
|---|---:|---:|
| L2 = 1e-5 | 0.8883 | 0.8835 |
| L2 = 1e-4 | 0.8965 | 0.8894 |
| L2 = 1e-3 | 0.8495 | 0.8335 |
| L2 = 1e-2 | 0.6777 | 0.5766 |

Best L2 by Val F1: `1e-4`

#### 2B: Dropout

| Setting | Val Acc | Val F1 |
|---|---:|---:|
| Dropout = 0.1 | 0.9144 | 0.9099 |
| Dropout = 0.25 | 0.8898 | 0.8778 |
| Dropout = 0.4 | 0.8801 | 0.8602 |
| Dropout = 0.6 | 0.8453 | 0.8162 |

Best dropout by Val F1: `0.1`

#### 2C: Data Augmentation

| Setting | Val Acc | Val F1 |
|---|---:|---:|
| Aug L1 | 0.9094 | 0.9059 |
| Aug L2 | 0.9043 | 0.8997 |
| Aug L3 | 0.8846 | 0.8765 |
| Aug L4 | 0.8447 | 0.8231 |

Best augmentation by Val F1: `Aug L1`

#### 2D: Combined Settings

| Setting | Val Acc | Val F1 |
|---|---:|---:|
| L2 = 1e-4 + Dropout = 0.1 | 0.8994 | 0.8947 |
| L2 = 1e-4 + Aug L1 | 0.8818 | 0.8721 |
| Dropout = 0.1 + Aug L1 | 0.9091 | 0.9032 |
| L2 + Dropout + Aug (all best) | 0.8909 | 0.8744 |

Best combined setting by Val F1: `Dropout = 0.1 + Aug L1`

RGB conclusions:
- The strongest RGB Part 2 setting on Arch C was dropout 0.1 with Val F1 = 0.9099.
- This improved over the Part 1 Arch C baseline (0.8857) by +0.0242 F1.
- Mild augmentation also helped, but not as much as dropout.
- Combining the individually best settings did not exceed the best dropout-only result.

## Cross-Modal Takeaways for Report 1

1. Larger CNNs consistently improved validation performance in both modalities.
2. RGB achieved the best raw Part 1 result with Arch D (Val F1 = 0.9185), outperforming IR Arch D (Val F1 = 0.8792).
3. IR benefited more strongly from augmentation than RGB in Part 2.
4. RGB responded best to light dropout, while IR responded best to moderate geometric augmentation.
5. In both modalities, "combine all individually best settings" did not necessarily produce the strongest result. This is worth highlighting as evidence that regularization effects are not simply additive.

## Suggested Report Language

Use wording along these lines:

- "For both modalities, increasing CNN capacity improved validation accuracy and weighted F1, with the largest architecture (Arch D) achieving the best Part 1 performance."
- "For RGB, Arch D produced the strongest overall validation result, while Arch C was selected for Part 2 because it exhibited a more moderate train-validation gap."
- "For IR, augmentation was the most effective regularization strategy, and augmentation level 2 improved validation F1 from 0.8792 to 0.9227."
- "For RGB, light dropout (0.1) was the strongest Part 2 setting on the selected base architecture, improving validation F1 from 0.8857 to 0.9099."
- "Combining the individually best regularization settings did not consistently improve performance, suggesting interaction effects between regularization techniques."

## Files to Cite in the Report

### Part 1 Figures
- `figures/part1_ir_architectures.png`
- `figures/part1_rgb_architectures.png`

### Part 2 Figures
- `figures/part2a_ir_l2.png`
- `figures/part2b_ir_dropout.png`
- `figures/part2c_ir_augmentation.png`
- `figures/part2d_ir_combined.png`
- `figures/part2a_rgb_l2.png`
- `figures/part2b_rgb_dropout.png`
- `figures/part2c_rgb_augmentation.png`
- `figures/part2d_rgb_combined.png`
