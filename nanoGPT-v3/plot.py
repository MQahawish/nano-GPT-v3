import matplotlib.pyplot as plt
import pandas as pd
import re
import argparse
import os
from typing import List, Dict, Any

def parse_eval_line(line: str) -> Dict[str, Any]:
    """Parse a line from eval.log"""
    # Only parse lines that start with "eval - WARNING"
    if not line.startswith("eval - WARNING"):
        return None
        
    try:
        step_match = re.search(r'step\s+(\d+)', line)
        if not step_match:
            return None
        
        step = int(step_match.group(1))
        
        # Extract all metrics
        train_match = re.search(r'train loss\s+([\d.]+)', line)
        val_match = re.search(r'val loss\s+([\d.]+)', line)
        lr_match = re.search(r'lr\s+([\d.]+)', line)
        mfu_match = re.search(r'mfu\s+([\d.]+)%', line)
        
        return {
            'step': step,
            'train_loss': float(train_match.group(1)) if train_match else None,
            'val_loss': float(val_match.group(1)) if val_match else None,
            'lr': float(lr_match.group(1)) if lr_match else None,
            'mfu': float(mfu_match.group(1)) if mfu_match else None
        }
    except Exception as e:
        print(f"Warning: Could not parse eval line: {line}")
        print(f"Error: {str(e)}")
        return None

def parse_training_line(line: str) -> Dict[str, Any]:
    """Parse a line from training.log"""
    try:
        # Parse any line that contains 'iter' and 'loss'
        iter_match = re.search(r'iter\s+(\d+)', line)
        if not iter_match:
            return None
        
        step = int(iter_match.group(1))
        
        # Extract training metrics
        loss_match = re.search(r'loss\s+([\d.]+)', line)
        time_match = re.search(r'time\s+([\d.]+)ms', line)
        mfu_match = re.search(r'mfu\s+([\d.]+)%', line)
        
        return {
            'step': step,
            'loss': float(loss_match.group(1)) if loss_match else None,
            'time_ms': float(time_match.group(1)) if time_match else None,
            'mfu': float(mfu_match.group(1)) if mfu_match else None
        }
    except Exception as e:
        print(f"Warning: Could not parse training line: {line}")
        print(f"Error: {str(e)}")
        return None

def plot_eval_metrics(log_file: str, output_file: str = None) -> None:
    """Plot evaluation metrics from eval.log"""
    metrics = []
    with open(log_file, 'r') as f:
        for line in f:
            parsed = parse_eval_line(line)
            if parsed:
                metrics.append(parsed)
    
    if not metrics:
        print("No valid eval metrics found in log")
        return
        
    df = pd.DataFrame(metrics)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Evaluation Metrics', fontsize=16)
    
    # Plot losses
    ax = axes[0,0]
    ax.plot(df['step'], df['train_loss'], label='Train Loss')
    ax.plot(df['step'], df['val_loss'], label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot learning rate
    ax = axes[0,1]
    ax.plot(df['step'], df['lr'])
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Learning Rate')
    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Plot MFU
    ax = axes[1,0]
    ax.plot(df['step'], df['mfu'])
    ax.set_title('Model Flops Utilization')
    ax.set_xlabel('Steps')
    ax.set_ylabel('MFU (%)')
    ax.grid(True)
    
    # Empty subplot
    axes[1,1].axis('off')
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to {output_file}")
    else:
        plt.show()
    plt.close()

def plot_training_metrics(log_file: str, output_file: str = None) -> None:
    """Plot training metrics from training.log"""
    metrics = []
    with open(log_file, 'r') as f:
        for line in f:
            parsed = parse_training_line(line)
            if parsed:
                metrics.append(parsed)
    
    if not metrics:
        print("No valid training metrics found in log")
        return
        
    df = pd.DataFrame(metrics)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Plot training loss
    ax = axes[0,0]
    ax.plot(df['step'], df['loss'], label='Training Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot time per iteration
    ax = axes[0,1]
    ax.plot(df['step'], df['time_ms'])
    ax.set_title('Time per Iteration')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Time (ms)')
    ax.grid(True)
    
    # Plot MFU
    ax = axes[1,0]
    ax.plot(df['step'], df['mfu'])
    ax.set_title('Model Flops Utilization')
    ax.set_xlabel('Steps')
    ax.set_ylabel('MFU (%)')
    ax.grid(True)
    
    # Empty subplot
    axes[1,1].axis('off')
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {output_file}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from log file')
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        return
    
    # Determine which type of log file it is based on filename
    if args.log_file.endswith('eval.log'):
        plot_eval_metrics(args.log_file, args.output)
    else:
        plot_training_metrics(args.log_file, args.output)

if __name__ == '__main__':
    main()