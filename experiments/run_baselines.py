#!/usr/bin/env python3
"""
Master orchestration script for CCDance baseline reproduction.
Runs all 5 baselines × 2 tasks × 5 seeds = 50 experiments in parallel across 4 GPUs.

Usage:
  python run_baselines.py --all                    # Run everything
  python run_baselines.py --baseline usdl          # Run single baseline
  python run_baselines.py --task classification    # Run classification only
  python run_baselines.py --gpus 0,1               # Use specific GPUs
  python run_baselines.py --status                 # Check experiment status
  python run_baselines.py --aggregate              # Aggregate results
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

BASELINE_ROOT = Path("/home/doudou/software/emc_results/experiments/baselines")
RESULTS_ROOT = Path("/home/doudou/software/emc_results/experiments/results_summary")

BASELINES = ["usdl", "core", "vl_transformer", "levit_hybrid", "graph_transformer"]
TASKS = ["classification", "generation"]
SEEDS = [42, 123, 456, 789, 1024]


def run_experiment(baseline, task, gpu_id, seed=None):
    """Run a single experiment."""
    script = BASELINE_ROOT / baseline / task / "train.py"
    if not script.exists():
        print(f"ERROR: Script not found: {script}")
        return False

    cmd = [
        sys.executable, str(script),
        "--gpu", str(gpu_id),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    log_dir = BASELINE_ROOT / baseline / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{task}_{timestamp}.log"

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"  Running: {' '.join(cmd)}  [GPU {gpu_id}]  Log: {log_file}")

    with open(log_file, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Baseline: {baseline}, Task: {task}, GPU: {gpu_id}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Start: {datetime.now()}\n")
        f.write(f"{'='*60}\n\n")
        f.flush()

        try:
            proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                  text=True, timeout=14400)  # 4 hour timeout
            if proc.returncode == 0:
                print(f"    ✓ Completed successfully")
                return True
            else:
                print(f"    ✗ Failed with return code {proc.returncode}")
                return False
        except subprocess.TimeoutExpired:
            print(f"    ✗ Timeout (>4 hours)")
            f.write("\nTIMEOUT: Experiment exceeded 4 hours\n")
            return False
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return False


def run_baseline_sequential(baseline, gpu_id):
    """Run both tasks for a baseline sequentially on one GPU."""
    results = {}

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"  Baseline: {baseline}, Task: {task}, GPU: {gpu_id}")
        print(f"{'='*60}")

        success = run_experiment(baseline, task, gpu_id)
        results[f"{baseline}_{task}"] = "completed" if success else "failed"

    return results


def run_all_parallel(gpu_ids):
    """Run all baselines in parallel across available GPUs.
    Each baseline gets one GPU; classification then generation run sequentially per GPU.
    """
    active_gpus = [int(g) for g in gpu_ids.split(",")]
    n_gpus = len(active_gpus)

    # Assign baselines to GPUs in round-robin
    assignments = {}
    for i, baseline in enumerate(BASELINES):
        gpu = active_gpus[i % n_gpus]
        if gpu not in assignments:
            assignments[gpu] = []
        assignments[gpu].append(baseline)

    print(f"\nGPU Assignments:")
    for gpu, baselines in assignments.items():
        print(f"  GPU {gpu}: {', '.join(baselines)}")

    # Run baselines assigned to each GPU sequentially
    # We'll use subprocess to run each GPU's work in parallel
    processes = []
    all_results = {}

    for gpu, baselines in assignments.items():
        # For each GPU, run all assigned baselines sequentially
        for baseline in baselines:
            for task in TASKS:
                script = BASELINE_ROOT / baseline / task / "train.py"
                if script.exists():
                    print(f"Launching: {baseline}/{task} on GPU {gpu}")
                    cmd = [sys.executable, str(script), "--gpu", str(gpu)]
                    log_dir = BASELINE_ROOT / baseline / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_file = log_dir / f"{task}_{datetime.now():%Y%m%d_%H%M%S}.log"

                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

                    with open(log_file, 'w') as f:
                        f.write(f"Baseline: {baseline}, Task: {task}, GPU: {gpu}\n")
                        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                        processes.append((f"{baseline}_{task}", proc, log_file))

    # Wait for all processes
    print(f"\nRunning {len(processes)} experiments...")
    for name, proc, log_file in processes:
        print(f"  Waiting for {name}...")
        proc.wait()
        status = "✓" if proc.returncode == 0 else "✗"
        all_results[name] = "completed" if proc.returncode == 0 else "failed"
        print(f"    {status} {name} (log: {log_file})")

    return all_results


def check_status():
    """Check status of all experiments."""
    print("\nExperiment Status:")
    print("=" * 80)

    for baseline in BASELINES:
        cls_agg = BASELINE_ROOT / baseline / "classification" / "aggregated_results.json"
        gen_agg = BASELINE_ROOT / baseline / "generation" / "aggregated_results.json"

        cls_status = "✓ Done" if cls_agg.exists() else "○ Pending"
        gen_status = "✓ Done" if gen_agg.exists() else "○ Pending"

        print(f"  {baseline:25s}  Classification: {cls_status:12s}  Generation: {gen_status:12s}")

        # Show seed-level progress
        for task in TASKS:
            completed_seeds = 0
            for seed in SEEDS:
                res_file = BASELINE_ROOT / baseline / task / f"seed_{seed}" / "results.json"
                if res_file.exists():
                    completed_seeds += 1
            if completed_seeds > 0:
                print(f"    {'':25s}  {task}: {completed_seeds}/{len(SEEDS)} seeds completed")

    # Check shared data
    pose_cache = Path("/home/doudou/software/emc_results/experiments/results/pose_sequences.pkl")
    print(f"\n  Pose cache: {'✓ Exists' if pose_cache.exists() else '✗ Missing (will be created)'}")


def main():
    parser = argparse.ArgumentParser(description="CCDance Baseline Reproduction Suite")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--baseline", type=str, help="Run specific baseline")
    parser.add_argument("--task", type=str, choices=TASKS, help="Run specific task")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="GPU IDs to use")
    parser.add_argument("--status", action="store_true", help="Check experiment status")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially on single GPU")
    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.aggregate:
        from results_summary.aggregate_results import (build_classification_table,
                                                        build_generation_table,
                                                        save_table_csv, save_table_md,
                                                        generate_summary_json)
        cls_h, cls_r = build_classification_table()
        gen_h, gen_r = build_generation_table()
        save_table_csv(cls_h, cls_r, "classification_results.csv")
        save_table_md(cls_h, cls_r, "classification_results.md",
                      "CCDance Classification Results")
        save_table_csv(gen_h, gen_r, "generation_results.csv")
        save_table_md(gen_h, gen_r, "generation_results.md",
                      "CCDance Generation Results")
        generate_summary_json()
        print("Results aggregated successfully!")
        return

    if args.sequential:
        # Run single baseline sequentially
        baseline = args.baseline or "all"
        gpu = args.gpus.split(",")[0]
        if baseline == "all":
            for bl in BASELINES:
                run_baseline_sequential(bl, gpu)
        else:
            run_baseline_sequential(baseline, gpu)
        return

    if args.all:
        results = run_all_parallel(args.gpus)
        print(f"\n{'='*60}")
        print("All experiments complete!")
        print(f"{'='*60}")
        for name, status in results.items():
            print(f"  {name}: {status}")

        # Auto-aggregate
        print("\nAggregating results...")
        subprocess.run([sys.executable, __file__, "--aggregate"])
        return

    if args.baseline:
        baseline = args.baseline
        task = args.task or "classification"
        gpu = args.gpus.split(",")[0]
        print(f"Running {baseline}/{task} on GPU {gpu}")
        run_experiment(baseline, task, gpu)
        return

    # Default: show status
    check_status()


if __name__ == '__main__':
    main()
