#!/usr/bin/env python3
"""
Traffic Light Detection - Training Pipeline
Phase 4 Implementation: Model Training, Validation, and Evaluation
"""

import os
import sys
import json
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class TrafficLightTrainer:
    """
    Comprehensive training pipeline for traffic light detection using YOLO
    """
    
    def __init__(self, config_path: str = "traffic_lights.yaml", 
                 experiment_name: str = None,
                 model_size: str = "yolo11n"):
        """
        Initialize the training pipeline
        
        Args:
            config_path: Path to YOLO dataset configuration file
            experiment_name: Custom experiment name (auto-generated if None)
            model_size: YOLO model size (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
        """
        self.config_path = Path(config_path)
        self.model_size = model_size
        
        # Load dataset configuration
        self.config = self._load_config()
        
        # Setup experiment tracking
        self.experiment_name = experiment_name or f"tld_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = Path(f"experiments/{self.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.model = None
        self.training_results = None
        self.best_metrics = {}
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Training hyperparameters
        self.training_config = self._setup_training_config()
        
    def _load_config(self) -> Dict:
        """Load YOLO dataset configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate configuration
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")
                
        return config
    
    def _setup_logging(self):
        """Setup logging for the training pipeline"""
        log_file = self.experiment_dir / "training.log"
        
        # Create logger
        self.logger = logging.getLogger(f"TrafficLightTrainer_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _setup_training_config(self) -> Dict:
        """Setup training hyperparameters optimized for traffic light detection"""
        
        # Base configuration optimized for small object detection (traffic lights)
        config = {
            # Training parameters
            'epochs': 300,              # Extended training for better convergence
            'patience': 50,             # Early stopping patience
            'batch': 16,                # Batch size (adjust based on GPU memory)
            'imgsz': 640,              # Image size (standard YOLO)
            
            # Optimization
            'lr0': 0.01,               # Initial learning rate
            'lrf': 0.01,               # Final learning rate fraction
            'momentum': 0.937,         # SGD momentum
            'weight_decay': 0.0005,    # Optimizer weight decay
            'warmup_epochs': 3.0,      # Warmup epochs
            'warmup_momentum': 0.8,    # Warmup initial momentum
            'warmup_bias_lr': 0.1,     # Warmup initial bias lr
            
            # Data augmentation (traffic light optimized)
            'hsv_h': 0.015,            # HSV-Hue augmentation (conservative)
            'hsv_s': 0.7,              # HSV-Saturation augmentation
            'hsv_v': 0.4,              # HSV-Value augmentation
            'degrees': 10,             # Image rotation (+/- deg)
            'translate': 0.1,          # Image translation (+/- fraction)
            'scale': 0.5,              # Image scale (+/- gain)
            'shear': 0.0,              # Image shear (+/- deg) - disabled for small objects
            'perspective': 0.0,        # Image perspective (+/- fraction) - disabled
            'flipud': 0.0,             # Image flip up-down (disabled)
            'fliplr': 0.5,             # Image flip left-right
            'mosaic': 1.0,             # Image mosaic (probability)
            'mixup': 0.1,              # Image mixup (probability)
            'copy_paste': 0.0,         # Segment copy-paste (disabled)
            
            # Loss function weights
            'box': 7.5,                # Box loss gain
            'cls': 0.5,                # Class loss gain
            'dfl': 1.5,                # Distribution focal loss gain
            
            # Validation and saving
            'save_period': 10,         # Save checkpoint every N epochs  
            'val': True,               # Validate during training
            'plots': True,             # Generate training plots
            'save': True,              # Save checkpoints
            
            # Advanced options
            'overlap_mask': True,      # Use overlap mask for training
            'mask_ratio': 4,           # Mask downsample ratio
            'dropout': 0.0,            # Use dropout regularization
            'cos_lr': False,           # Use cosine LR scheduler
            'close_mosaic': 10,        # Disable mosaic augmentation for final epochs
            
            # Model configuration
            'pretrained': True,        # Use pretrained weights
            'verbose': True,           # Verbose output
            'seed': 42,                # Random seed for reproducibility
            'deterministic': True,     # Use deterministic algorithms
            'single_cls': False,       # Train as single-class dataset
            'rect': False,             # Rectangular training
            'resume': False,           # Resume training from last checkpoint
            'nosave': False,           # Only save final checkpoint
            'noval': False,            # Only validate final epoch
            'noautoanchor': False,     # Disable AutoAnchor
            'noplots': False,          # Disable plotting
            'evolve': None,            # Evolve hyperparameters for N generations
            'bucket': '',              # Google Cloud Storage bucket
            'cache': False,            # Cache images for faster training
            'image_weights': False,    # Use weighted image selection for training
            'device': str(self.device), # Training device
            'multi_scale': False,      # Vary img-size +/- 50%
            'optimizer': 'SGD',        # Optimizer (Adam, AdamW, NAdam, RAdam, RMSProp, SGD)
            'sync_bn': False,          # Use SyncBatchNorm
            'workers': 8,              # Max dataloader workers
            'project': str(self.experiment_dir.parent),  # Project directory
            'name': self.experiment_name,                # Experiment name
            'exist_ok': True,          # Whether existing project/name is ok
            'half': False,             # Use FP16 half-precision training
            'dnn': False,              # Use OpenCV DNN for ONNX inference
            'amp': True,               # Automatic Mixed Precision training
        }
        
        return config
    
    def load_model(self, weights_path: Optional[str] = None) -> YOLO:
        """
        Load YOLO model for training
        
        Args:
            weights_path: Path to custom weights (uses pretrained if None)
            
        Returns:
            Loaded YOLO model
        """
        if weights_path and Path(weights_path).exists():
            self.logger.info(f"Loading custom weights from: {weights_path}")
            model = YOLO(weights_path)
        else:
            # Use pretrained weights based on model size
            pretrained_weights = f"{self.model_size}.pt"
            self.logger.info(f"Loading pretrained weights: {pretrained_weights}")
            model = YOLO(pretrained_weights)
        
        self.model = model
        return model
    
    def validate_dataset(self) -> bool:
        """
        Validate dataset before training
        
        Returns:
            True if dataset is valid, False otherwise
        """
        self.logger.info("Validating dataset...")
        
        # Check paths
        dataset_path = Path(self.config['path'])
        train_path = dataset_path / self.config['train']
        val_path = dataset_path / self.config['val']
        
        if not dataset_path.exists():
            self.logger.error(f"Dataset path does not exist: {dataset_path}")
            return False
            
        if not train_path.exists():
            self.logger.error(f"Training path does not exist: {train_path}")
            return False
            
        if not val_path.exists():
            self.logger.error(f"Validation path does not exist: {val_path}")
            return False
        
        # Check for images and labels
        train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
        val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
        
        train_labels_path = dataset_path / "labels" / "train"
        val_labels_path = dataset_path / "labels" / "val"
        
        train_labels = list(train_labels_path.glob("*.txt")) if train_labels_path.exists() else []
        val_labels = list(val_labels_path.glob("*.txt")) if val_labels_path.exists() else []
        
        self.logger.info(f"Training images: {len(train_images)}")
        self.logger.info(f"Training labels: {len(train_labels)}")
        self.logger.info(f"Validation images: {len(val_images)}")
        self.logger.info(f"Validation labels: {len(val_labels)}")
        
        # Basic validation
        if len(train_images) == 0:
            self.logger.error("No training images found")
            return False
            
        if len(val_images) == 0:
            self.logger.error("No validation images found")
            return False
        
        # Check class count
        if self.config['nc'] != 25:
            self.logger.warning(f"Expected 25 classes, found {self.config['nc']}")
        
        self.logger.info("Dataset validation completed successfully")
        return True
    
    def train(self, resume: bool = False) -> Dict:
        """
        Train the traffic light detection model
        
        Args:
            resume: Whether to resume from last checkpoint
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.load_model()
        
        # Validate dataset
        if not self.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        self.logger.info(f"Starting training with {self.model_size} model")
        self.logger.info(f"Training configuration: {json.dumps(self.training_config, indent=2)}")
        
        # Update resume flag
        if resume:
            self.training_config['resume'] = resume
        
        # Start training
        start_time = time.time()
        
        try:
            # Train the model
            results = self.model.train(
                data=str(self.config_path),
                **self.training_config
            )
            
            self.training_results = results
            training_time = time.time() - start_time
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save training summary
            self._save_training_summary(results, training_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def validate(self, weights_path: Optional[str] = None) -> Dict:
        """
        Validate the trained model
        
        Args:
            weights_path: Path to model weights (uses best if None)
            
        Returns:
            Validation results dictionary
        """
        if weights_path:
            model = YOLO(weights_path)
        elif self.model:
            model = self.model
        else:
            # Try to find best weights
            best_weights = self.experiment_dir / "weights" / "best.pt"
            if best_weights.exists():
                model = YOLO(str(best_weights))
            else:
                raise ValueError("No model weights available for validation")
        
        self.logger.info("Starting model validation...")
        
        # Validate the model
        results = model.val(
            data=str(self.config_path),
            imgsz=self.training_config['imgsz'],
            batch=self.training_config['batch'],
            device=self.device,
            plots=True,
            save_json=True
        )
        
        # Log validation metrics
        self._log_validation_metrics(results)
        
        return results
    
    def _save_training_summary(self, results, training_time: float):
        """Save comprehensive training summary"""
        
        summary = {
            'experiment_name': self.experiment_name,
            'model_size': self.model_size,
            'dataset_config': self.config,
            'training_config': self.training_config,
            'training_time_seconds': training_time,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add results if available
        if hasattr(results, 'results_dict'):
            summary['final_metrics'] = results.results_dict
        
        # Save to JSON
        summary_path = self.experiment_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved to: {summary_path}")
    
    def _log_validation_metrics(self, results):
        """Log detailed validation metrics"""
        
        metrics = {
            'mAP50': getattr(results, 'box.map50', 0),
            'mAP50-95': getattr(results, 'box.map', 0),
            'Precision': getattr(results, 'box.mp', 0),
            'Recall': getattr(results, 'box.mr', 0),
        }
        
        self.logger.info("Validation Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # Save metrics
        metrics_path = self.experiment_dir / "validation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def export_model(self, format: str = 'onnx', weights_path: Optional[str] = None):
        """
        Export trained model to different formats
        
        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
            weights_path: Path to model weights
        """
        if weights_path:
            model = YOLO(weights_path)
        elif self.model:
            model = self.model
        else:
            best_weights = self.experiment_dir / "weights" / "best.pt"
            if best_weights.exists():
                model = YOLO(str(best_weights))
            else:
                raise ValueError("No model weights available for export")
        
        self.logger.info(f"Exporting model to {format} format...")
        
        export_path = model.export(format=format, imgsz=self.training_config['imgsz'])
        
        self.logger.info(f"Model exported to: {export_path}")
        return export_path
    
    def create_training_plots(self):
        """Create comprehensive training visualization plots"""
        
        results_dir = self.experiment_dir / "runs" / "detect" / self.experiment_name
        
        if not results_dir.exists():
            self.logger.warning(f"Results directory not found: {results_dir}")
            return
        
        # YOLO automatically generates plots, but we can create additional custom ones
        self.logger.info("Training plots created automatically by YOLO")
        self.logger.info(f"Check results in: {results_dir}")
    
    def get_experiment_summary(self) -> Dict:
        """Get comprehensive experiment summary"""
        
        summary = {
            'experiment_name': self.experiment_name,
            'model_size': self.model_size,
            'dataset_path': str(self.config_path),
            'experiment_dir': str(self.experiment_dir),
            'device': str(self.device),
            'training_config': self.training_config,
        }
        
        # Add training results if available
        if self.training_results:
            summary['training_completed'] = True
            summary['training_results'] = str(self.training_results)
        else:
            summary['training_completed'] = False
        
        return summary

# Utility functions for training pipeline management

def setup_training_environment():
    """Setup optimal training environment"""
    
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optimize for performance
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        # Enable automatic mixed precision if supported
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def get_available_models() -> List[str]:
    """Get list of available YOLO model sizes"""
    return ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']

def estimate_training_time(num_images: int, epochs: int, model_size: str, device: str) -> float:
    """
    Estimate training time based on dataset size and model complexity
    
    Args:
        num_images: Total number of training images
        epochs: Number of training epochs
        model_size: YOLO model size
        device: Training device (cuda/cpu)
        
    Returns:
        Estimated training time in hours
    """
    
    # Base times per image per epoch (in seconds)
    base_times = {
        'yolo11n': 0.001,   # Nano - fastest
        'yolo11s': 0.002,   # Small
        'yolo11m': 0.004,   # Medium  
        'yolo11l': 0.008,   # Large
        'yolo11x': 0.015,   # Extra Large - slowest
    }
    
    base_time = base_times.get(model_size, 0.004)
    
    # Adjust for device
    if device == 'cpu':
        base_time *= 10  # CPU is much slower
    
    # Calculate total time
    total_seconds = num_images * epochs * base_time
    total_hours = total_seconds / 3600
    
    return total_hours

if __name__ == "__main__":
    # Setup training environment
    setup_training_environment()
    
    # Initialize trainer
    trainer = TrafficLightTrainer(
        config_path="traffic_lights.yaml",
        model_size="yolo11n",  # Start with nano model for faster training
        experiment_name="traffic_light_detection_v1"
    )
    
    # Print experiment summary
    summary = trainer.get_experiment_summary()
    print(f"Experiment Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Train the model
    print("\nStarting training...")
    results = trainer.train()
    
    # Validate the model
    print("\nStarting validation...")
    val_results = trainer.validate()
    
    # Export model
    print("\nExporting model...")
    export_path = trainer.export_model(format='onnx')
    
    print(f"\nTraining pipeline completed!")
    print(f"Experiment directory: {trainer.experiment_dir}")