#!/usr/bin/env python3
"""
Run All Echo Pretraining Experiments

Runs JEPA, MAE, and Supervised baseline in sequence.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINING_DIR = os.path.join(SCRIPT_DIR, "pretraining")
DEFAULT_JEPA_CONFIG = os.path.join(SCRIPT_DIR, "configs", "echo_vit_small.yaml")


def run_experiment(script_name, args_list=None):
    """Run a pretraining script."""
    script_path = os.path.join(PRETRAINING_DIR, script_name)
    cmd = [sys.executable, script_path]
    if args_list:
        cmd.extend(args_list)
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().isoformat()}")
    print('='*60 + '\n')
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"WARNING: {script_name} exited with code {result.returncode}")
    else:
        print(f"\n{script_name} completed successfully!")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run all Echo pretraining experiments')
    parser.add_argument('--skip_jepa', action='store_true', help='Skip JEPA training')
    parser.add_argument('--skip_mae', action='store_true', help='Skip MAE training')
    parser.add_argument('--skip_supervised', action='store_true', help='Skip Supervised baseline')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ECHO SEGMENTATION PRETRAINING EXPERIMENTS")
    print("="*60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    results = {}
    
    # 1. JEPA (config-driven; epochs/batch_size come from the YAML config,
    # not CLI flags -- jepa_pretrain.py only accepts --config/--device/--resume/--max_images)
    if not args.skip_jepa:
        results['jepa'] = run_experiment('jepa_pretrain.py', [
            '--config', DEFAULT_JEPA_CONFIG
        ])
    
    # 2. MAE
    if not args.skip_mae:
        results['mae'] = run_experiment('mae_pretrain.py', [
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size)
        ])
    
    # 3. Supervised baseline (no training needed)
    if not args.skip_supervised:
        results['supervised'] = run_experiment('supervised_pretrain.py')
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, code in results.items():
        status = "SUCCESS" if code == 0 else f"FAILED (code {code})"
        print(f"  {name}: {status}")
    print(f"End time: {datetime.now().isoformat()}")
    print("="*60)


if __name__ == '__main__':
    main()
