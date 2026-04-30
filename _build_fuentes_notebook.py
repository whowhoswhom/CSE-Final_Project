"""One-off builder for CSE547_FinalProject_Report2_Fuentes.ipynb.

Produces a fully self-contained notebook that loads pre-computed JSON / CSV / PNG
artifacts and displays them as visible tables and figures. No imports of
report2_pipeline, no torch, no exec, no training, no inference.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NB_PATH = Path("CSE547_FinalProject_Report2_Fuentes.ipynb")


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


cells: list[dict] = []

# 0. Header
cells.append(md(
    "# CSE 547 Final Project \N{EM DASH} Report 2\n"
    "\n"
    "**Author:** Jose Fuentes  \n"
    "**Course:** CSE 547 \n\n"
    "This notebook documents the experimental work for **Parts 3, 4, 5, and 6** of the final project "
    "(RGB / IR object recognition on paired video scenes). Parts 1 and 2 are treated as validated baselines; "
    "they are referenced only for comparison against the Part 3, 4, and 6 models. All training was completed "
    "previously on a CUDA workstation; this notebook loads the cached JSON summaries, prediction CSVs, and "
    "result figures from the `figures/` and `manifests/` folders and renders them inline.\n"
    "\n"
    "The notebook is self-contained: it uses only the Python standard library, `pandas`, `matplotlib`, and "
    "`IPython.display`. It does not import any project module, does not load model checkpoints, and does "
    "not retrain or re-infer anything.\n"
))

# 1. Setup header + imports + helpers
cells.append(md("## 1. Environment and helpers"))

cells.append(code(
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from IPython.display import Image, Markdown, display\n"
    "\n"
    "%matplotlib inline\n"
    "\n"
    "pd.set_option(\"display.max_columns\", 20)\n"
    "pd.set_option(\"display.width\", 140)\n"
    "pd.set_option(\"display.float_format\", lambda v: f\"{v:.4f}\")\n"
    "\n"
    "FIGURES_DIR = Path(\"figures\")\n"
    "MANIFESTS_DIR = Path(\"manifests\")\n"
))

cells.append(code(
    "def load_json(name):\n"
    "    \"\"\"Load a JSON file from the figures/ folder.\"\"\"\n"
    "    with (FIGURES_DIR / name).open(\"r\", encoding=\"utf-8\") as fh:\n"
    "        return json.load(fh)\n"
    "\n"
    "\n"
    "def show_figure(name, caption=None):\n"
    "    \"\"\"Render a PNG from figures/ with an optional markdown caption.\"\"\"\n"
    "    if caption:\n"
    "        display(Markdown(f\"*{caption}*\"))\n"
    "    display(Image(filename=str(FIGURES_DIR / name)))\n"
    "\n"
    "\n"
    "def best_of(run, key):\n"
    "    \"\"\"Pick the max value of a per-epoch history list (Part 3/4 runs store arrays).\"\"\"\n"
    "    series = run.get(key)\n"
    "    if series is None:\n"
    "        return None\n"
    "    if isinstance(series, list):\n"
    "        return max(series) if series else None\n"
    "    return series\n"
))

# 2. Dataset context
cells.append(md(
    "## 2. Dataset context\n"
    "\n"
    "The project uses paired RGB and thermal (IR) patches extracted from driving-scene videos across 8 "
    "object classes. Train / validation splits are defined at the **video** level to prevent leakage, and "
    "every manifest row stores the patch path, class label, source video, and split assignment. The table "
    "below is computed directly from the two manifest CSVs.\n"
))

cells.append(code(
    "rgb_m = pd.read_csv(MANIFESTS_DIR / \"rgb_manifest.csv\", usecols=[\"modality\", \"class_name\", \"split\"])\n"
    "ir_m  = pd.read_csv(MANIFESTS_DIR / \"ir_manifest.csv\",  usecols=[\"modality\", \"class_name\", \"split\"])\n"
    "\n"
    "\n"
    "def manifest_summary(df, name):\n"
    "    return {\n"
    "        \"modality\": name,\n"
    "        \"total_patches\": len(df),\n"
    "        \"train_patches\": int((df[\"split\"] == \"train\").sum()),\n"
    "        \"val_patches\":   int((df[\"split\"] == \"val\").sum()),\n"
    "        \"n_classes\":     df[\"class_name\"].nunique(),\n"
    "    }\n"
    "\n"
    "\n"
    "dataset_summary = pd.DataFrame([\n"
    "    manifest_summary(rgb_m, \"RGB\"),\n"
    "    manifest_summary(ir_m,  \"IR\"),\n"
    "])\n"
    "dataset_summary\n"
))

# 3. Part 3 — RGB VGG16
cells.append(md(
    "## 3. Part 3 \N{EM DASH} RGB VGG16 transfer learning\n"
    "\n"
    "**Setup.** The Part 3 experiment reuses the ImageNet-pretrained VGG16 convolutional base with a two-"
    "layer dense classifier on top. Three freeze levels were trained: **F1** (freeze all conv blocks), "
    "**F2** (freeze everything except block5), and **F3** (freeze only blocks 1\N{EN DASH}3). The four-panel "
    "figure below compares train / validation accuracy and train / validation F1 across all three.\n"
))

cells.append(code(
    "part3 = load_json(\"part3_rgb_vgg16_results.json\")\n"
    "\n"
    "part3_table = pd.DataFrame([\n"
    "    {\n"
    "        \"label\":         run[\"label\"],\n"
    "        \"description\":   run[\"description\"],\n"
    "        \"freeze_level\":  run[\"freeze_level\"],\n"
    "        \"trainable_params\": run[\"trainable_params\"],\n"
    "        \"best_train_acc\": best_of(run, \"train_acc\"),\n"
    "        \"best_val_acc\":   best_of(run, \"val_acc\"),\n"
    "        \"best_train_f1\":  best_of(run, \"train_f1\"),\n"
    "        \"best_val_f1\":    best_of(run, \"val_f1\"),\n"
    "    }\n"
    "    for run in part3\n"
    "])\n"
    "part3_table = part3_table.sort_values(\"best_val_f1\", ascending=False).reset_index(drop=True)\n"
    "part3_table\n"
))

cells.append(code(
    "show_figure(\n"
    "    \"part3_rgb_vgg16_freeze.png\",\n"
    "    caption=\"Part 3 \N{EM DASH} VGG16 RGB freeze-level comparison (train/val accuracy and F1 across F1, F2, F3).\",\n"
    ")\n"
))

# 4. Part 4 — IR AE
cells.append(md(
    "## 4. Part 4 \N{EM DASH} IR convolutional autoencoder + dense classifier\n"
    "\n"
    "**Setup.** Six convolutional autoencoder architectures were trained on IR patches (varying conv depth, "
    "filter counts, and bottleneck dimension). Each frozen encoder was then used as a feature extractor for "
    "a two-layer dense classifier, under **two regularization settings** (a light dropout baseline and a "
    "stronger dropout + weight-decay variant), yielding **12 total configurations**. The four-panel figure "
    "reports train / validation accuracy and F1 across all twelve labeled options with different colors and "
    "a legend.\n"
))

cells.append(code(
    "part4 = load_json(\"part4_ir_ae_results.json\")\n"
    "\n"
    "part4_table = pd.DataFrame([\n"
    "    {\n"
    "        \"label\":          run[\"label\"],\n"
    "        \"ae_config\":      run[\"ae_config\"],\n"
    "        \"filters\":        \"-\".join(str(f) for f in run[\"filters\"]),\n"
    "        \"bottleneck_dim\": run[\"bottleneck_dim\"],\n"
    "        \"reg_config\":     run[\"reg_config\"],\n"
    "        \"dropout\":        run[\"dropout\"],\n"
    "        \"weight_decay\":   run[\"weight_decay\"],\n"
    "        \"best_train_acc\": best_of(run, \"train_acc\"),\n"
    "        \"best_val_acc\":   best_of(run, \"val_acc\"),\n"
    "        \"best_train_f1\":  best_of(run, \"train_f1\"),\n"
    "        \"best_val_f1\":    best_of(run, \"val_f1\"),\n"
    "    }\n"
    "    for run in part4\n"
    "])\n"
    "part4_table = part4_table.sort_values(\"best_val_f1\", ascending=False).reset_index(drop=True)\n"
    "part4_table\n"
))

cells.append(code(
    "show_figure(\n"
    "    \"part4_ir_autoencoder_12_options.png\",\n"
    "    caption=\"Part 4 \N{EM DASH} 12 AE + classifier combinations (train/val accuracy and F1 across labeled options).\",\n"
    ")\n"
))

# 5. Part 6 — refinement
cells.append(md(
    "## 5. Part 6 \N{EM DASH} bounded refinement for blind-test readiness\n"
    "\n"
    "**Setup.** The strongest Part 3 RGB candidate received a short mild-augmentation refinement pass. "
    "On the IR side, none of the Part 4 autoencoder + classifier options came within 0.02 validation F1 "
    "of the Part 1 / Part 2 supervised IR CNN (Arch D), so IR refinement was skipped and the stronger "
    "supervised CNN was carried forward as the final IR model. All selection uses validation data only; "
    "no blind-test labels were consulted.\n"
))

cells.append(code(
    "part6 = load_json(\"part6_refinement_results.json\")\n"
    "\n"
    "rgb_refined = part6[\"rgb_refined\"]\n"
    "ir_refined  = part6[\"ir_refined\"]\n"
    "\n"
    "part6_rows = []\n"
    "if rgb_refined is not None:\n"
    "    part6_rows.append({\n"
    "        \"sensor\":           \"RGB\",\n"
    "        \"refined_label\":    rgb_refined[\"label\"],\n"
    "        \"best_train_acc\":   rgb_refined.get(\"best_train_acc\", best_of(rgb_refined, \"train_acc\")),\n"
    "        \"best_val_acc\":     rgb_refined.get(\"best_val_acc\",   best_of(rgb_refined, \"val_acc\")),\n"
    "        \"best_train_f1\":    rgb_refined.get(\"best_train_f1\",  best_of(rgb_refined, \"train_f1\")),\n"
    "        \"best_val_f1\":      rgb_refined.get(\"best_val_f1\",    best_of(rgb_refined, \"val_f1\")),\n"
    "    })\n"
    "if ir_refined is not None:\n"
    "    part6_rows.append({\n"
    "        \"sensor\":           \"IR\",\n"
    "        \"refined_label\":    ir_refined[\"label\"],\n"
    "        \"best_train_acc\":   ir_refined.get(\"best_train_acc\", best_of(ir_refined, \"train_acc\")),\n"
    "        \"best_val_acc\":     ir_refined.get(\"best_val_acc\",   best_of(ir_refined, \"val_acc\")),\n"
    "        \"best_train_f1\":    ir_refined.get(\"best_train_f1\",  best_of(ir_refined, \"train_f1\")),\n"
    "        \"best_val_f1\":      ir_refined.get(\"best_val_f1\",    best_of(ir_refined, \"val_f1\")),\n"
    "    })\n"
    "part6_table = pd.DataFrame(part6_rows)\n"
    "\n"
    "print(\"Part 6 notes:\")\n"
    "for n in part6.get(\"notes\", []):\n"
    "    print(\" -\", n)\n"
    "part6_table\n"
))

# 6. Final model selection
cells.append(md(
    "## 6. Final model selection\n"
    "\n"
    "Final RGB and IR models are chosen by **validation F1 only**, drawing candidates from Part 1 baselines, "
    "Part 2 regularized variants, Part 3 (RGB transfer learning), Part 4 (IR autoencoders), and Part 6 "
    "refinement. The candidates table below lists every contender considered; the two-row headline table "
    "shows the final picks used for the blind-test submission.\n"
))

cells.append(code(
    "selection = load_json(\"report2_final_model_selection.json\")\n"
    "\n"
    "\n"
    "def format_train(value):\n"
    "    \"\"\"Render train metrics that were not logged (stored as 0 / None) as 'n/a'.\"\"\"\n"
    "    if value is None or value == 0.0 or value == 0:\n"
    "        return \"n/a\"\n"
    "    return value\n"
    "\n"
    "\n"
    "candidate_cols = [\"modality\", \"source\", \"label\", \"arch\", \"kind\", \"best_train_acc\", \"best_val_acc\", \"best_train_f1\", \"best_val_f1\"]\n"
    "candidates = pd.DataFrame(\n"
    "    selection[\"rgb_candidates\"] + selection[\"ir_candidates\"]\n"
    ")[candidate_cols]\n"
    "candidates = candidates.sort_values([\"modality\", \"best_val_f1\"], ascending=[True, False]).reset_index(drop=True)\n"
    "for col in (\"best_train_acc\", \"best_train_f1\"):\n"
    "    candidates[col] = candidates[col].apply(format_train)\n"
    "candidates\n"
))

cells.append(code(
    "final_rgb = selection[\"final_rgb\"]\n"
    "final_ir  = selection[\"final_ir\"]\n"
    "\n"
    "headline = pd.DataFrame([\n"
    "    {\n"
    "        \"sensor\":         \"RGB\",\n"
    "        \"final_label\":    final_rgb[\"label\"],\n"
    "        \"best_train_acc\": format_train(final_rgb.get(\"best_train_acc\", best_of(final_rgb, \"train_acc\"))),\n"
    "        \"best_val_acc\":   final_rgb.get(\"best_val_acc\",   best_of(final_rgb, \"val_acc\")),\n"
    "        \"best_train_f1\":  format_train(final_rgb.get(\"best_train_f1\",  best_of(final_rgb, \"train_f1\"))),\n"
    "        \"best_val_f1\":    final_rgb.get(\"best_val_f1\",    best_of(final_rgb, \"val_f1\")),\n"
    "    },\n"
    "    {\n"
    "        \"sensor\":         \"IR\",\n"
    "        \"final_label\":    final_ir[\"label\"],\n"
    "        \"best_train_acc\": format_train(final_ir.get(\"best_train_acc\", best_of(final_ir, \"train_acc\"))),\n"
    "        \"best_val_acc\":   final_ir.get(\"best_val_acc\",   best_of(final_ir, \"val_acc\")),\n"
    "        \"best_train_f1\":  format_train(final_ir.get(\"best_train_f1\",  best_of(final_ir, \"train_f1\"))),\n"
    "        \"best_val_f1\":    final_ir.get(\"best_val_f1\",    best_of(final_ir, \"val_f1\")),\n"
    "    },\n"
    "])\n"
    "headline\n"
))

# 7. Part 5 — error + sensor analysis
cells.append(md(
    "## 7. Part 5 \N{EM DASH} error analysis and sensor comparison\n"
    "\n"
    "Using the final RGB and IR models, every validation sample was scored and stored in "
    "`figures/part5_rgb_predictions.csv` and `figures/part5_ir_predictions.csv`. The analysis below is "
    "built directly from those CSVs and the cached Part 5 summary JSON; no inference is re-run.\n"
))

cells.append(code(
    "rgb_preds = pd.read_csv(FIGURES_DIR / \"part5_rgb_predictions.csv\")\n"
    "ir_preds  = pd.read_csv(FIGURES_DIR / \"part5_ir_predictions.csv\")\n"
    "\n"
    "overall = pd.DataFrame([\n"
    "    {\n"
    "        \"sensor\":      \"RGB\",\n"
    "        \"n_val\":       len(rgb_preds),\n"
    "        \"val_acc\":     rgb_preds[\"correct\"].mean(),\n"
    "        \"n_miss\":      int((~rgb_preds[\"correct\"]).sum()),\n"
    "    },\n"
    "    {\n"
    "        \"sensor\":      \"IR\",\n"
    "        \"n_val\":       len(ir_preds),\n"
    "        \"val_acc\":     ir_preds[\"correct\"].mean(),\n"
    "        \"n_miss\":      int((~ir_preds[\"correct\"]).sum()),\n"
    "    },\n"
    "])\n"
    "overall\n"
))

cells.append(code(
    "def miss_by_class(df, sensor):\n"
    "    grouped = df[~df[\"correct\"]].groupby(\"true_class\").size().rename(f\"{sensor}_misclassified\")\n"
    "    return grouped.to_frame().reset_index()\n"
    "\n"
    "\n"
    "rgb_miss = miss_by_class(rgb_preds, \"RGB\").sort_values(\"RGB_misclassified\", ascending=False).reset_index(drop=True)\n"
    "ir_miss  = miss_by_class(ir_preds,  \"IR\") .sort_values(\"IR_misclassified\",  ascending=False).reset_index(drop=True)\n"
    "\n"
    "miss_table = rgb_miss.merge(ir_miss, on=\"true_class\", how=\"outer\").fillna(0).astype({\"RGB_misclassified\": int, \"IR_misclassified\": int})\n"
    "miss_table\n"
))

cells.append(code(
    "show_figure(\n"
    "    \"part5_rgb_misclassified_by_class.png\",\n"
    "    caption=\"Part 5 \N{EM DASH} up to two RGB misclassifications per class (highest-confidence errors).\",\n"
    ")\n"
))

cells.append(code(
    "show_figure(\n"
    "    \"part5_ir_misclassified_by_class.png\",\n"
    "    caption=\"Part 5 \N{EM DASH} up to two IR misclassifications per class (highest-confidence errors).\",\n"
    ")\n"
))

cells.append(code(
    "analysis = load_json(\"part5_analysis_summary.json\")\n"
    "\n"
    "paired = pd.DataFrame([\n"
    "    {\"scenario\": \"RGB correct, IR wrong\", \"count\": analysis[\"rgb_correct_ir_wrong_count\"]},\n"
    "    {\"scenario\": \"IR correct, RGB wrong\", \"count\": analysis[\"ir_correct_rgb_wrong_count\"]},\n"
    "])\n"
    "paired\n"
))

cells.append(code(
    "show_figure(\n"
    "    \"part5_rgb_correct_ir_wrong.png\",\n"
    "    caption=\"Paired validation scenes where the RGB model was correct and the IR model was wrong.\",\n"
    ")\n"
))

cells.append(code(
    "show_figure(\n"
    "    \"part5_ir_correct_rgb_wrong.png\",\n"
    "    caption=\"Paired validation scenes where the IR model was correct and the RGB model was wrong.\",\n"
    ")\n"
))

# 8. Justification + integrity
cells.append(md(
    "### Justification for misclassifications and sensor reliability\n"
    "\n"
    "RGB generally wins on classes that depend on **color and fine texture** \N{EM DASH} traffic signs, "
    "signal lights, and small objects whose shape cues are weak in a thermal view. IR pulls ahead in "
    "**low-light or visually cluttered** scenes where visible contrast collapses but thermal contrast "
    "against the background remains strong (vehicles, pedestrians at night, vehicles partially behind "
    "foliage). The largest single IR error cluster is **bus** (436 misclassifications), consistent with "
    "the IR sensor struggling to disambiguate large vehicles of similar thermal signature; RGB, in "
    "contrast, spreads its errors more evenly across classes. The paired-scene grids above illustrate "
    "these trade-offs concretely.\n"
))

cells.append(md(
    "## 8. Integrity and blind-test statement\n"
    "\n"
    "Final models were selected exclusively on **validation F1**. No blind-test labels were consulted "
    "during training, hyperparameter selection, Part 5 analysis, or Part 6 refinement. The two rows in "
    "the headline table in Section 6 (**RGB VGG16 refined** and **IR CNN Arch D + Aug L2**) are the "
    "models submitted for the April-24 blind-test evaluation. No withheld-test images participated in "
    "any training or selection step.\n"
))

# 9. References
cells.append(md(
    "## 9. References\n"
    "\n"
    "- Written report: `CSE547_FinalProject_Report2_Fuentes.pdf` (5 pages).\n"
    "- Training / evaluation / plotting source code: `report2_pipeline.py` in this project folder (not "
    "  imported by this notebook, but available for inspection).\n"
    "- Cached result artifacts used by every cell above: JSON summaries and PNG figures under `figures/`, "
    "  prediction CSVs `figures/part5_rgb_predictions.csv` and `figures/part5_ir_predictions.csv`, and "
    "  manifests `manifests/rgb_manifest.csv` / `manifests/ir_manifest.csv`.\n"
    "- The 10-minute video walkthrough is linked on Blackboard.\n"
))


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "final-exam-cse547",
        "language": "python",
        "name": "final-exam-cse547",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

with NB_PATH.open("w", encoding="utf-8") as fh:
    nbf.write(nb, fh)

print(f"wrote {NB_PATH} with {len(cells)} cells")
