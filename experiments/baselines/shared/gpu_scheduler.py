"""
Multi-GPU parallel job scheduler for CCDance baseline experiments.
Manages 4 GPUs, queues experiments, prevents memory overflow.
"""
import os
import sys
import json
import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime


# GPU configuration
N_GPUS = 4
GPU_MEMORY_GB = 48
BASE_DIR = Path("/home/doudou/software/emc_results/experiments/baselines")


class GPUJob:
    """Represents a single experiment job."""

    def __init__(self, name, script_path, gpu_id, args=None, mem_required_gb=8):
        self.name = name
        self.script_path = script_path
        self.gpu_id = gpu_id
        self.args = args or []
        self.mem_required_gb = mem_required_gb
        self.status = 'queued'  # queued, running, completed, failed
        self.start_time = None
        self.end_time = None
        self.process = None
        self.log_file = None


class GPUScheduler:
    """Manages multiple GPU jobs with FIFO scheduling."""

    def __init__(self, n_gpus=N_GPUS, gpu_memory=GPU_MEMORY_GB):
        self.n_gpus = n_gpus
        self.gpu_memory = gpu_memory
        self.jobs = []
        self.queue = []
        self.gpu_available_mem = [gpu_memory] * n_gpus
        self.running_jobs = {i: None for i in range(n_gpus)}

    def add_job(self, name, script_path, gpu_id=None, args=None, mem_required=8):
        """Add a job to the queue."""
        job = GPUJob(name, script_path, gpu_id, args, mem_required)
        self.queue.append(job)
        return job

    def _can_schedule(self, job, gpu_id):
        """Check if a job can be scheduled on a specific GPU."""
        if self.running_jobs[gpu_id] is not None:
            return False
        if job.mem_required_gb > self.gpu_available_mem[gpu_id]:
            return False
        return True

    def _find_gpu(self, job):
        """Find the best GPU for a job."""
        if job.gpu_id is not None and self._can_schedule(job, job.gpu_id):
            return job.gpu_id

        # Find GPU with most available memory
        best_gpu = None
        best_mem = -1
        for gpu_id in range(self.n_gpus):
            if self._can_schedule(job, gpu_id) and self.gpu_available_mem[gpu_id] > best_mem:
                best_gpu = gpu_id
                best_mem = self.gpu_available_mem[gpu_id]

        return best_gpu

    def schedule_all(self):
        """Schedule all queued jobs across available GPUs."""
        scheduled = []
        remaining = []

        for job in self.queue:
            if job.status != 'queued':
                remaining.append(job)
                continue

            gpu_id = self._find_gpu(job)
            if gpu_id is not None:
                job.gpu_id = gpu_id
                job.status = 'running'
                job.start_time = datetime.now()
                self.running_jobs[gpu_id] = job
                self.gpu_available_mem[gpu_id] -= job.mem_required_gb
                scheduled.append(job)
            else:
                remaining.append(job)

        self.queue = remaining
        return scheduled

    def run_job(self, job):
        """Execute a single job (non-blocking)."""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(job.gpu_id)

        log_dir = Path(job.script_path).parent.parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        job.log_file = log_dir / f"{job.name}_{datetime.now():%Y%m%d_%H%M%S}.log"

        cmd = [sys.executable, job.script_path] + job.args

        with open(job.log_file, 'w') as f:
            f.write(f"Job: {job.name}\n")
            f.write(f"GPU: {job.gpu_id}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start: {job.start_time}\n")
            f.write(f"{'='*60}\n\n")
            f.flush()

            try:
                job.process = subprocess.Popen(
                    cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                    text=True
                )
            except Exception as e:
                f.write(f"\nERROR: {e}\n")
                job.status = 'failed'

    def run_all_queued(self):
        """Run all queued jobs (blocks until all complete)."""
        # Phase 1: Schedule all jobs
        while self.queue:
            scheduled = self.schedule_all()
            if not scheduled:
                # Check if any running jobs have completed
                self._check_completed()
                if not self.queue:
                    break
                time.sleep(10)
                continue

            for job in scheduled:
                print(f"Starting {job.name} on GPU {job.gpu_id}")
                self.run_job(job)

            # Wait for jobs to complete
            self._wait_for_completion()

        print("\nAll jobs completed!")

    def _check_completed(self):
        """Check for completed jobs and free GPU resources."""
        for gpu_id, job in list(self.running_jobs.items()):
            if job is None:
                continue
            if job.process is not None:
                ret = job.process.poll()
                if ret is not None:
                    job.end_time = datetime.now()
                    job.status = 'completed' if ret == 0 else 'failed'
                    self.gpu_available_mem[gpu_id] += job.mem_required_gb
                    self.running_jobs[gpu_id] = None
                    print(f"  Job {job.name} on GPU {gpu_id} "
                          f"{'completed' if ret == 0 else 'failed'} "
                          f"in {(job.end_time - job.start_time).total_seconds():.0f}s")

    def _wait_for_completion(self):
        """Wait for any running job to complete."""
        while all(j is not None for j in self.running_jobs.values()):
            self._check_completed()
            if any(j is None for j in self.running_jobs.values()):
                break
            time.sleep(10)

    def get_status(self):
        """Get current status of all jobs."""
        status = {
            'queued': len([j for j in self.queue if j.status == 'queued']),
            'running': len([j for j in self.running_jobs.values() if j is not None]),
            'completed': len([j for j in self.jobs if j.status == 'completed']),
            'failed': len([j for j in self.jobs if j.status == 'failed']),
        }
        for gpu_id, job in self.running_jobs.items():
            if job is not None:
                status[f'gpu_{gpu_id}'] = job.name
            else:
                status[f'gpu_{gpu_id}'] = 'idle'
        return status


def generate_experiment_matrix():
    """Generate the full experiment matrix: 5 baselines × 2 tasks × 5 seeds."""
    baselines = ['usdl', 'core', 'vl_transformer', 'levit_hybrid', 'graph_transformer']
    tasks = ['classification', 'generation']

    experiments = []
    for baseline in baselines:
        for task in tasks:
            script_path = BASE_DIR / baseline / task / 'train.py'
            experiment = {
                'baseline': baseline,
                'task': task,
                'name': f'{baseline}_{task}',
                'script_path': str(script_path),
                'gpu_mem': 8,  # default GPU memory requirement
            }
            # Adjust for larger models
            if baseline == 'levit_hybrid':
                experiment['gpu_mem'] = 16
            elif baseline == 'vl_transformer':
                experiment['gpu_mem'] = 12

            experiments.append(experiment)

    return experiments


def run_experiment_suite(experiments, parallel=True):
    """Run the full experiment suite."""
    scheduler = GPUScheduler()

    for exp in experiments:
        scheduler.add_job(
            name=exp['name'],
            script_path=exp['script_path'],
            args=['--gpu', '0'],  # Will be overridden by scheduler
            mem_required=exp.get('gpu_mem', 8),
        )

    if parallel:
        scheduler.run_all_queued()
    else:
        # Sequential execution
        for job in scheduler.queue:
            job.gpu_id = 0
            scheduler.gpu_available_mem[0] = GPU_MEMORY_GB
            scheduler.running_jobs[0] = job
            scheduler.run_job(job)
            job.process.wait()
            scheduler._check_completed()

    # Print summary
    print("\n" + "=" * 60)
    print("Experiment Suite Summary")
    print("=" * 60)
    for job in scheduler.jobs:
        duration = ""
        if job.start_time and job.end_time:
            secs = (job.end_time - job.start_time).total_seconds()
            duration = f" ({secs:.0f}s)"
        print(f"  {job.name}: {job.status}{duration}")


if __name__ == '__main__':
    # Quick test
    experiments = generate_experiment_matrix()
    for exp in experiments:
        print(f"{exp['name']}: {exp['script_path']} (GPU mem: {exp['gpu_mem']}GB)")
