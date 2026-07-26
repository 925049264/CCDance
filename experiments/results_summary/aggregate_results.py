"""
Aggregate results from all baseline experiments into unified tables.
Generates CSV + Markdown format summary tables.
"""
import json
import csv
from pathlib import Path

BASELINE_ROOT = Path("/home/doudou/software/emc_results/experiments/baselines")
OUTPUT_DIR = Path("/home/doudou/software/emc_results/experiments/results_summary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINES = {
    "usdl": "USDL (Tang et al. CVPR 2020)",
    "core": "CoRe (Yu et al. ICCV 2021)",
    "vl_transformer": "VL-Transformer (Chen, SciRep 2025)",
    "levit_hybrid": "LeViT-Hybrid (Wang, SciRep 2025)",
    "graph_transformer": "Graph-Transformer (Han et al., SciRep 2026)",
}

TASKS = ["classification", "generation"]


def load_baseline_results(baseline, task):
    """Load aggregated results for a baseline+task combination.
    Searches multiple possible paths due to varied output directory structures.
    """
    # Possible paths where results might be stored
    possible_paths = [
        BASELINE_ROOT / baseline / task / "aggregated_results.json",
        BASELINE_ROOT / baseline / "results" / task / "aggregated_results.json",
        BASELINE_ROOT / baseline / "results" / "aggregated_results.json",
    ]

    for result_path in possible_paths:
        if result_path.exists():
            with open(result_path) as f:
                return json.load(f)

    # If no aggregated, try to aggregate from per-seed results
    seed_results = []
    for seed_dir in sorted((BASELINE_ROOT / baseline / "results").glob("seed_*")):
        # Check multiple depths for results.json
        for subpath in ["results.json", "finetune/results.json", "classification/results.json"]:
            rp = seed_dir / subpath
            if rp.exists():
                with open(rp) as f:
                    data = json.load(f)
                if "test_metrics" in data:
                    seed_results.append(data["test_metrics"])
                elif "accuracy" in data:
                    seed_results.append(data)
                break

    if seed_results:
        from shared.metrics import compute_classification_metrics_mean_std
        return compute_classification_metrics_mean_std(seed_results)

    return None


def build_classification_table():
    """Build the main classification results table."""
    rows = []
    header = ["Model", "Accuracy", "Accuracy Std", "Macro-F1", "Macro-F1 Std", "QWK", "QWK Std"]

    # Add existing baselines (from the original paper)
    existing_results = {
        "Random Baseline": {"accuracy": 0.333, "accuracy_std": 0.0, "macro_f1": 0.333, "macro_f1_std": 0.0, "qwk": 0.0, "qwk_std": 0.0},
        "XGBoost (handcrafted)": {"accuracy": 0.434, "accuracy_std": 0.046, "macro_f1": 0.431, "macro_f1_std": 0.049, "qwk": 0.262, "qwk_std": 0.039},
        "SVM (handcrafted)": {"accuracy": 0.296, "accuracy_std": 0.018, "macro_f1": 0.270, "macro_f1_std": 0.029, "qwk": 0.098, "qwk_std": 0.024},
        "PoseLSTM": {"accuracy": 0.343, "accuracy_std": 0.075, "macro_f1": 0.324, "macro_f1_std": 0.065, "qwk": 0.097, "qwk_std": 0.133},
        "ST-GCN": {"accuracy": 0.343, "accuracy_std": 0.064, "macro_f1": 0.321, "macro_f1_std": 0.058, "qwk": 0.143, "qwk_std": 0.107},
        "Pose Transformer": {"accuracy": 0.332, "accuracy_std": 0.009, "macro_f1": 0.166, "macro_f1_std": 0.003, "qwk": 0.000, "qwk_std": 0.0},
        "Two-Stage (DanceMVP)": {"accuracy": 0.341, "accuracy_std": 0.015, "macro_f1": 0.330, "macro_f1_std": 0.0, "qwk": 0.126, "qwk_std": 0.0},
    }

    for name, res in existing_results.items():
        rows.append([name, res["accuracy"], res["accuracy_std"],
                     res["macro_f1"], res["macro_f1_std"],
                     res["qwk"], res["qwk_std"]])

    # Add reproduced baselines
    for bid, name in BASELINES.items():
        res = load_baseline_results(bid, "classification")
        if res:
            rows.append([
                name,
                res.get("accuracy", "N/A"),
                res.get("accuracy_std", "N/A"),
                res.get("macro_f1", "N/A"),
                res.get("macro_f1_std", "N/A"),
                res.get("qwk", "N/A"),
                res.get("qwk_std", "N/A"),
            ])
        else:
            rows.append([name, "pending", "pending", "pending", "pending", "pending", "pending"])

    return header, rows


def build_generation_table():
    """Build the generation results table."""
    rows = []
    header = ["Model", "BLEU-1", "BLEU-4", "ROUGE-L", "BERTScore"]

    existing_results = {
        "LSTM Seq2Seq": {"bleu1": 0.095, "bleu4": 0.012, "rouge_l": 0.068, "bertscore": 0.132},
        "Multimodal Transformer": {"bleu1": 0.009, "bleu4": 0.002, "rouge_l": 0.015, "bertscore": 0.065},
        "Two-Stage (DanceMVP)": {"bleu1": 0.052, "bleu4": 0.008, "rouge_l": 0.045, "bertscore": 0.098},
        "Human Agreement (upper bound)": {"bleu1": 0.47, "bleu4": "N/A", "rouge_l": 0.52, "bertscore": "N/A"},
    }

    for name, res in existing_results.items():
        rows.append([name, res["bleu1"], res["bleu4"], res["rouge_l"], res["bertscore"]])

    for bid, name in BASELINES.items():
        res = load_baseline_results(bid, "generation")
        if res:
            rows.append([
                name,
                res.get("bleu1", "N/A"),
                res.get("bleu4", "N/A"),
                res.get("rouge_l", "N/A"),
                res.get("bertscore", "N/A"),
            ])
        else:
            rows.append([name, "pending", "pending", "pending", "pending"])

    return header, rows


def save_table_csv(header, rows, filename):
    """Save a table as CSV."""
    with open(OUTPUT_DIR / filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_table_md(header, rows, filename, title):
    """Save a table as Markdown."""
    with open(OUTPUT_DIR / filename, 'w') as f:
        f.write(f"# {title}\n\n")
        # Header
        f.write("| " + " | ".join(str(h) for h in header) + " |\n")
        f.write("|" + "|".join("---" for _ in header) + "|\n")
        # Rows
        for row in rows:
            # Format numbers
            formatted = []
            for cell in row:
                if isinstance(cell, float):
                    formatted.append(f"{cell:.3f}")
                else:
                    formatted.append(str(cell))
            f.write("| " + " | ".join(formatted) + " |\n")


def generate_summary_json():
    """Generate a comprehensive JSON summary of all results."""
    summary = {
        "dataset": "CCDance",
        "total_videos": 175,
        "total_genres": 22,
        "grade_distribution": {"A": 58, "B": 58, "C": 59},
        "evaluation_protocols": ["random_split_70_15_15"],
        "n_seeds": 5,
        "seeds": [42, 123, 456, 789, 1024],
        "classification_metrics": ["accuracy", "macro_f1", "qwk", "ece"],
        "generation_metrics": ["bleu1", "bleu2", "bleu4", "rouge_l", "bertscore"],
        "baselines": {},
    }

    for bid, name in BASELINES.items():
        cls_res = load_baseline_results(bid, "classification")
        gen_res = load_baseline_results(bid, "generation")
        summary["baselines"][bid] = {
            "name": name,
            "classification": cls_res,
            "generation": gen_res,
        }

    # Also load individual seed results
    for bid in BASELINES:
        cls_dir = BASELINE_ROOT / bid / "classification"
        gen_dir = BASELINE_ROOT / bid / "generation"

        for seed in [42, 123, 456, 789, 1024]:
            seed_res = cls_dir / f"seed_{seed}" / "results.json"
            if seed_res.exists():
                with open(seed_res) as f:
                    data = json.load(f)
                if "per_seed_results" not in summary["baselines"][bid]:
                    summary["baselines"][bid]["per_seed_results"] = {}
                summary["baselines"][bid]["per_seed_results"][f"seed_{seed}"] = data

    with open(OUTPUT_DIR / "master_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    print("Aggregating baseline experiment results...")
    print("=" * 60)

    # Classification table
    cls_header, cls_rows = build_classification_table()
    print("\n### Classification Results ###")
    print(" | ".join(cls_header))
    for row in cls_rows:
        fmt = [f"{c:.3f}" if isinstance(c, float) else str(c) for c in row]
        print(" | ".join(fmt))

    save_table_csv(cls_header, cls_rows, "classification_results.csv")
    save_table_md(cls_header, cls_rows, "classification_results.md",
                  "CCDance Dance Quality Grade Classification Results")

    # Generation table
    gen_header, gen_rows = build_generation_table()
    print("\n### Generation Results ###")
    print(" | ".join(gen_header))
    for row in gen_rows:
        fmt = [f"{c:.3f}" if isinstance(c, float) else str(c) for c in row]
        print(" | ".join(fmt))

    save_table_csv(gen_header, gen_rows, "generation_results.csv")
    save_table_md(gen_header, gen_rows, "generation_results.md",
                  "CCDance Dance Quality Comment Generation Results")

    # Full JSON summary
    summary = generate_summary_json()
    print(f"\nResults saved to {OUTPUT_DIR}/")

    # Check status
    completed = sum(1 for bid in BASELINES
                    if load_baseline_results(bid, "classification") is not None)
    print(f"Classification experiments completed: {completed}/{len(BASELINES)}")
    completed_gen = sum(1 for bid in BASELINES
                        if load_baseline_results(bid, "generation") is not None)
    print(f"Generation experiments completed: {completed_gen}/{len(BASELINES)}")
