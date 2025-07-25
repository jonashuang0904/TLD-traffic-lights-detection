#!/usr/bin/env python3
"""
Traffic Light Detection - Model Training Execution Script
Phase 4 Implementation: Complete Training Pipeline with Advanced Monitoring
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Import our custom modules
from .training_pipeline import TrafficLightTrainer, setup_training_environment
from .training_config import TrafficLightTrainingConfig, create_default_configs

class TrainingMonitor:
    """
    Advanced training monitor with real-time logging and visualization
    """
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'map50': [],
            'map50_95': [],
            'precision': [],
            'recall': [],
            'learning_rate': [],
            'gpu_memory': []
        }
        
        # Setup monitoring logger
        self.logger = self._setup_logger()
        
        # Start time
        self.start_time = time.time()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup detailed monitoring logger"""
        log_file = self.experiment_dir / "training_monitor.log"
        
        logger = logging.getLogger("TrainingMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler  
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_training_start(self, config: Dict):
        """Log training start information"""
        self.logger.info("=" * 80)
        self.logger.info("TRAFFIC LIGHT DETECTION - TRAINING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Model: {config.get('model_size', 'unknown')}")
        self.logger.info(f"Epochs: {config.get('epochs', 'unknown')}")
        self.logger.info(f"Batch Size: {config.get('batch', 'unknown')}")
        self.logger.info(f"Learning Rate: {config.get('lr0', 'unknown')}")
        self.logger.info(f"Device: {config.get('device', 'unknown')}")
        self.logger.info("=" * 80)
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict):
        """Log metrics for current epoch"""
        
        # Update metrics history
        self.metrics_history['epoch'].append(epoch)
        
        # Extract and store metrics
        train_loss = metrics.get('train/box_loss', 0) + metrics.get('train/cls_loss', 0)
        val_loss = metrics.get('val/box_loss', 0) + metrics.get('val/cls_loss', 0)
        
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['map50'].append(metrics.get('metrics/mAP50(B)', 0))
        self.metrics_history['map50_95'].append(metrics.get('metrics/mAP50-95(B)', 0))
        self.metrics_history['precision'].append(metrics.get('metrics/precision(B)', 0))
        self.metrics_history['recall'].append(metrics.get('metrics/recall(B)', 0))
        self.metrics_history['learning_rate'].append(metrics.get('lr/pg0', 0))
        
        # GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.metrics_history['gpu_memory'].append(gpu_memory)
        else:
            self.metrics_history['gpu_memory'].append(0)
        
        # Log current metrics
        elapsed_time = time.time() - self.start_time
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f} | "
            f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f} | "
            f"Time: {elapsed_time/60:.1f}min"
        )
    
    def create_training_plots(self):
        """Create comprehensive training visualization plots"""
        
        if len(self.metrics_history['epoch']) < 2:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Traffic Light Detection - Training Progress', fontsize=16, fontweight='bold')
        
        epochs = self.metrics_history['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Val Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # mAP curves
        axes[0, 1].plot(epochs, self.metrics_history['map50'], label='mAP@0.5', color='green', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics_history['map50_95'], label='mAP@0.5:0.95', color='orange', linewidth=2)
        axes[0, 1].set_title('Mean Average Precision', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[0, 2].plot(epochs, self.metrics_history['precision'], label='Precision', color='purple', linewidth=2)
        axes[0, 2].plot(epochs, self.metrics_history['recall'], label='Recall', color='brown', linewidth=2)
        axes[0, 2].set_title('Precision & Recall', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.metrics_history['learning_rate'], color='red', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # GPU Memory Usage
        if max(self.metrics_history['gpu_memory']) > 0:
            axes[1, 1].plot(epochs, self.metrics_history['gpu_memory'], color='darkblue', linewidth=2)
            axes[1, 1].set_title('GPU Memory Usage', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'CPU Training', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Device: CPU', fontweight='bold')
        
        # Training Summary
        if len(epochs) > 0:
            best_map50 = max(self.metrics_history['map50'])
            best_map50_95 = max(self.metrics_history['map50_95'])
            final_loss = self.metrics_history['train_loss'][-1]
            
            summary_text = f"""
Training Summary:
• Total Epochs: {max(epochs)}
• Best mAP@0.5: {best_map50:.4f}
• Best mAP@0.5:0.95: {best_map50_95:.4f}
• Final Loss: {final_loss:.4f}
• Total Time: {(time.time() - self.start_time)/3600:.2f} hours
"""
            axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                           fontsize=12, verticalalignment='center', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            axes[1, 2].set_title('Training Summary', fontweight='bold')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.experiment_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to: {plot_path}")
    
    def save_metrics_history(self):
        """Save complete metrics history to file"""
        metrics_path = self.experiment_dir / "metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        self.logger.info(f"Metrics history saved to: {metrics_path}")
    
    def log_training_complete(self, results):
        """Log training completion information"""
        total_time = time.time() - self.start_time
        
        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Time: {total_time/3600:.2f} hours")
        self.logger.info(f"Final Metrics:")
        
        if len(self.metrics_history['map50']) > 0:
            self.logger.info(f"  Best mAP@0.5: {max(self.metrics_history['map50']):.4f}")
            self.logger.info(f"  Best mAP@0.5:0.95: {max(self.metrics_history['map50_95']):.4f}")
            self.logger.info(f"  Final Loss: {self.metrics_history['train_loss'][-1]:.4f}")
        
        self.logger.info("=" * 80)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Traffic Light Detection Model')
    
    parser.add_argument('--model', type=str, default='yolo11n',
                       choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                       help='YOLO model size')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom training configuration file')
    
    parser.add_argument('--preset', type=str, default='balanced',
                       choices=['fast', 'balanced', 'accuracy', 'production'],
                       help='Use preset configuration')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (overrides config)')
    
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    parser.add_argument('--experiment', type=str, default=None,
                       help='Custom experiment name')
    
    parser.add_argument('--data', type=str, default='traffic_lights.yaml',
                       help='Path to dataset configuration file')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (auto, cpu, cuda, etc.)')
    
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers')
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation, no training')
    
    parser.add_argument('--export', type=str, default=None,
                       choices=['onnx', 'torchscript', 'tflite', 'engine'],
                       help='Export model format after training')
    
    return parser.parse_args()

def main():
    """Main training execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup training environment
    setup_training_environment()
    
    print("🚦 Traffic Light Detection - Training Pipeline")
    print("=" * 60)
    
    # Create training configuration
    if args.config:
        config_manager = TrafficLightTrainingConfig(args.config)
        print(f"📝 Loaded custom configuration from: {args.config}")
    else:
        # Use preset configuration
        preset_configs = create_default_configs()
        if args.preset in preset_configs:
            config_manager = preset_configs[args.preset]
            print(f"📝 Using preset configuration: {args.preset}")
        else:
            config_manager = TrafficLightTrainingConfig()
            print(f"📝 Using default configuration")
    
    # Override configuration with command line arguments
    if args.model:
        config_manager.model.model_size = args.model
    if args.epochs:
        config_manager.training.epochs = args.epochs
    if args.batch:
        config_manager.data.batch_size = args.batch
    if args.lr:
        config_manager.optimizer.lr0 = args.lr
    if args.device != 'auto':
        config_manager.environment.device = args.device
    if args.workers:
        config_manager.data.workers = args.workers
    if args.resume:
        config_manager.environment.resume = True
    if args.experiment:
        config_manager.environment.name = args.experiment
    
    # Get optimized configuration
    training_config = config_manager.get_traffic_light_optimized_config()
    
    # Print configuration summary
    config_manager.print_config_summary()
    
    # Initialize trainer
    trainer = TrafficLightTrainer(
        config_path=args.data,
        experiment_name=config_manager.environment.name,
        model_size=config_manager.model.model_size
    )
    
    # Initialize training monitor
    monitor = TrainingMonitor(trainer.experiment_dir)
    
    try:
        if args.validate_only:
            # Validation only mode
            print("\n🔍 Running validation only...")
            results = trainer.validate()
            monitor.logger.info("Validation completed successfully")
            
        else:
            # Full training pipeline
            print(f"\n🏋️ Starting training with {config_manager.model.model_size} model...")
            
            # Load model
            trainer.load_model()
            
            # Log training start
            monitor.log_training_start(training_config)
            
            # Train the model
            results = trainer.train(resume=args.resume)
            
            # Log training completion
            monitor.log_training_complete(results)
            
            # Create training plots
            monitor.create_training_plots()
            
            # Save metrics
            monitor.save_metrics_history()
            
            # Validate the trained model
            print("\n🔍 Running final validation...")
            val_results = trainer.validate()
            
            # Export model if requested
            if args.export:
                print(f"\n📦 Exporting model to {args.export} format...")
                export_path = trainer.export_model(format=args.export)
                print(f"✅ Model exported to: {export_path}")
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"📁 Results saved to: {trainer.experiment_dir}")
        
        # Print final summary
        summary = trainer.get_experiment_summary()
        print("\n📊 Experiment Summary:")
        print(f"  Experiment: {summary['experiment_name']}")
        print(f"  Model: {summary['model_size']}")
        print(f"  Device: {summary['device']}")
        print(f"  Directory: {summary['experiment_dir']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        monitor.logger.warning("Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        monitor.logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()