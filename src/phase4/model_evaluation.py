#!/usr/bin/env python3
"""
Traffic Light Detection - Model Evaluation and Validation System
Phase 4 Implementation: Comprehensive Model Assessment and Analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging

import torch
import cv2
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

class TrafficLightEvaluator:
    """
    Comprehensive evaluation system for traffic light detection models
    """
    
    def __init__(self, model_path: str, data_config: str, output_dir: str = "evaluation_results"):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to trained model weights
            data_config: Path to dataset configuration file
            output_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.data_config = Path(data_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Load dataset configuration
        self.dataset_config = self._load_dataset_config()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Class names from dataset config
        self.class_names = self.dataset_config.get('names', {})
        self.num_classes = self.dataset_config.get('nc', 25)
        
        # Results storage
        self.evaluation_results = {}
        self.validation_metrics = {}
        self.per_class_metrics = {}
        
    def _load_model(self) -> YOLO:
        """Load the trained YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model = YOLO(str(self.model_path))
        return model
    
    def _load_dataset_config(self) -> Dict:
        """Load dataset configuration"""
        import yaml
        
        if not self.data_config.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.data_config}")
        
        with open(self.data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_logger(self) -> logging.Logger:
        """Setup evaluation logger"""
        log_file = self.output_dir / "evaluation.log"
        
        logger = logging.getLogger("ModelEvaluator")
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
    
    def run_comprehensive_evaluation(self) -> Dict:
        """
        Run complete model evaluation pipeline
        
        Returns:
            Dictionary containing all evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        start_time = datetime.now()
        
        # 1. Basic validation metrics
        self.logger.info("Computing validation metrics...")
        validation_results = self.compute_validation_metrics()
        
        # 2. Per-class performance analysis
        self.logger.info("Analyzing per-class performance...")
        class_analysis = self.analyze_per_class_performance()
        
        # 3. Detection confidence analysis
        self.logger.info("Analyzing detection confidence...")
        confidence_analysis = self.analyze_detection_confidence()
        
        # 4. Speed and efficiency benchmarks
        self.logger.info("Running speed benchmarks...")
        speed_benchmarks = self.benchmark_inference_speed()
        
        # 5. Visualization and reporting
        self.logger.info("Creating evaluation visualizations...")
        self.create_evaluation_visualizations()
        
        # 6. Model analysis
        self.logger.info("Analyzing model architecture...")
        model_analysis = self.analyze_model_architecture()
        
        # Compile comprehensive results
        evaluation_time = (datetime.now() - start_time).total_seconds()
        
        comprehensive_results = {
            'evaluation_info': {
                'model_path': str(self.model_path),
                'data_config': str(self.data_config),
                'evaluation_time': evaluation_time,
                'timestamp': datetime.now().isoformat(),
                'num_classes': self.num_classes
            },
            'validation_metrics': validation_results,
            'per_class_analysis': class_analysis,
            'confidence_analysis': confidence_analysis,
            'speed_benchmarks': speed_benchmarks,
            'model_analysis': model_analysis
        }
        
        # Save comprehensive results
        self._save_evaluation_results(comprehensive_results)
        
        self.logger.info(f"Comprehensive evaluation completed in {evaluation_time:.2f} seconds")
        return comprehensive_results
    
    def compute_validation_metrics(self) -> Dict:
        """Compute detailed validation metrics"""
        
        # Run validation
        results = self.model.val(
            data=str(self.data_config),
            imgsz=640,
            batch=16,
            save_json=True,
            plots=True,
            verbose=False
        )
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'map_per_class': results.box.maps.tolist() if results.box.maps is not None else [],
            'fitness': float(results.fitness) if hasattr(results, 'fitness') else 0.0
        }
        
        # Additional computed metrics
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        self.validation_metrics = metrics
        return metrics
    
    def analyze_per_class_performance(self) -> Dict:
        """Analyze performance for each traffic light class"""
        
        class_analysis = {}
        
        # Get per-class mAP if available
        if 'map_per_class' in self.validation_metrics and self.validation_metrics['map_per_class']:
            map_per_class = self.validation_metrics['map_per_class']
            
            for class_id, class_name in self.class_names.items():
                if class_id < len(map_per_class):
                    class_analysis[class_name] = {
                        'class_id': class_id,
                        'mAP50_95': float(map_per_class[class_id]),
                        'performance_tier': self._classify_performance_tier(map_per_class[class_id])
                    }
        
        # Analyze class distribution and difficulty
        class_difficulty = self._analyze_class_difficulty()
        for class_name, analysis in class_analysis.items():
            if class_name in class_difficulty:
                analysis.update(class_difficulty[class_name])
        
        self.per_class_metrics = class_analysis
        return class_analysis
    
    def _classify_performance_tier(self, map_score: float) -> str:
        """Classify performance into tiers"""
        if map_score >= 0.8:
            return "Excellent"
        elif map_score >= 0.6:
            return "Good"
        elif map_score >= 0.4:
            return "Fair"
        elif map_score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _analyze_class_difficulty(self) -> Dict:
        """Analyze difficulty of each class based on various factors"""
        
        # Traffic light class difficulty analysis based on detection challenges
        difficulty_factors = {
            # Simple circle lights (easier to detect)
            'circle_green': {'difficulty': 'Easy', 'factors': ['Simple shape', 'High contrast']},
            'circle_red': {'difficulty': 'Easy', 'factors': ['Simple shape', 'High contrast']},
            'circle_yellow': {'difficulty': 'Easy', 'factors': ['Simple shape', 'High contrast']},
            'off': {'difficulty': 'Hard', 'factors': ['Low contrast', 'No distinctive color']},
            
            # Arrow lights (more complex shapes)
            'arrow_left_green': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Directional']},
            'arrow_right_green': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Directional']},
            'arrow_straight_green': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Directional']},
            
            # Red arrows (potentially harder due to color similarity)
            'arrow_left_red': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Color similarity']},
            'arrow_right_red': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Color similarity']},
            'arrow_straight_red': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Color similarity']},
            
            # Yellow arrows (medium difficulty)
            'arrow_left_yellow': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Medium contrast']},
            'arrow_right_yellow': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Medium contrast']},
            'arrow_straight_yellow': {'difficulty': 'Medium', 'factors': ['Complex shape', 'Medium contrast']},
            
            # Multi-color combinations (harder due to multiple states)
            'circle_red_yellow': {'difficulty': 'Hard', 'factors': ['Multi-color', 'State combination']},
            'arrow_left_red_yellow': {'difficulty': 'Hard', 'factors': ['Multi-color', 'Complex shape']},
            'arrow_right_red_yellow': {'difficulty': 'Hard', 'factors': ['Multi-color', 'Complex shape']},
            'arrow_straight_red_yellow': {'difficulty': 'Hard', 'factors': ['Multi-color', 'Complex shape']},
            
            # Complex multi-directional arrows (hardest)
            'arrow_straight_left_green': {'difficulty': 'Very Hard', 'factors': ['Multi-directional', 'Complex shape']},
            'arrow_straight_left_red': {'difficulty': 'Very Hard', 'factors': ['Multi-directional', 'Complex shape']},
            'arrow_straight_left_yellow': {'difficulty': 'Very Hard', 'factors': ['Multi-directional', 'Complex shape']},
            'arrow_straight_left_red_yellow': {'difficulty': 'Very Hard', 'factors': ['Multi-directional', 'Multi-color']},
            'arrow_straight_right_red': {'difficulty': 'Very Hard', 'factors': ['Multi-directional', 'Complex shape']},
            'arrow_straight_right_red_yellow': {'difficulty': 'Very Hard', 'factors': ['Multi-directional', 'Multi-color']},
            'arrow_straight_right_yellow': {'difficulty': 'Very Hard', 'factors': ['Multi-directional', 'Complex shape']},
            'arrow_straight_right_green': {'difficulty': 'Very Hard', 'factors': ['Multi-directional', 'Complex shape']},
        }
        
        return difficulty_factors
    
    def analyze_detection_confidence(self) -> Dict:
        """Analyze detection confidence distribution"""
        
        # This would require running inference on validation set
        # For now, return basic analysis structure
        confidence_analysis = {
            'mean_confidence': 0.0,
            'confidence_distribution': {
                'high_confidence': 0,  # > 0.8
                'medium_confidence': 0,  # 0.5 - 0.8
                'low_confidence': 0,  # < 0.5
            },
            'confidence_by_class': {},
            'recommended_threshold': 0.5
        }
        
        return confidence_analysis
    
    def benchmark_inference_speed(self) -> Dict:
        """Benchmark model inference speed"""
        
        # Create dummy input for speed testing
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Warm up
        for _ in range(10):
            _ = self.model(dummy_input, verbose=False)
        
        # Time inference
        times = []
        num_runs = 100
        
        for _ in range(num_runs):
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
                _ = self.model(dummy_input, verbose=False)
                end_time.record()
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time))
            else:
                import time
                start = time.time()
                _ = self.model(dummy_input, verbose=False)
                end = time.time()
                times.append((end - start) * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        times = np.array(times)
        
        speed_metrics = {
            'mean_inference_time_ms': float(np.mean(times)),
            'std_inference_time_ms': float(np.std(times)),
            'min_inference_time_ms': float(np.min(times)),
            'max_inference_time_ms': float(np.max(times)),
            'median_inference_time_ms': float(np.median(times)),
            'fps': 1000.0 / float(np.mean(times)),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        return speed_metrics
    
    def analyze_model_architecture(self) -> Dict:
        """Analyze model architecture and parameters"""
        
        # Get model info
        model_info = self.model.info(detailed=True, verbose=False)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        
        # Model size estimation (MB)
        param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
        
        architecture_analysis = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': float(param_size),
            'layers': len(list(self.model.model.modules())),
            'model_type': type(self.model.model).__name__,
        }
        
        return architecture_analysis
    
    def create_evaluation_visualizations(self):
        """Create comprehensive evaluation visualizations"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Metrics Overview Dashboard
        self._create_metrics_dashboard()
        
        # 2. Per-class Performance Chart
        self._create_per_class_chart()
        
        # 3. Performance vs Difficulty Analysis
        self._create_difficulty_analysis()
        
        # 4. Speed Benchmark Visualization
        self._create_speed_visualization()
        
        self.logger.info("Evaluation visualizations created successfully")
    
    def _create_metrics_dashboard(self):
        """Create main metrics dashboard"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Traffic Light Detection - Model Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        # Overall metrics
        metrics = self.validation_metrics
        metric_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics.get('mAP50', 0),
            metrics.get('mAP50_95', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        
        # Bar chart of main metrics
        bars = axes[0, 0].bar(metric_names, metric_values, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700', '#9370DB'])
        axes[0, 0].set_title('Overall Performance Metrics', fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance tier distribution
        if self.per_class_metrics:
            tiers = [info.get('performance_tier', 'Unknown') for info in self.per_class_metrics.values()]
            tier_counts = pd.Series(tiers).value_counts()
            
            colors = {'Excellent': '#2E8B57', 'Good': '#4169E1', 'Fair': '#FFD700', 
                     'Poor': '#FF6347', 'Very Poor': '#8B0000'}
            
            wedges, texts, autotexts = axes[0, 1].pie(tier_counts.values, labels=tier_counts.index,
                                                     autopct='%1.1f%%', startangle=90,
                                                     colors=[colors.get(tier, '#gray') for tier in tier_counts.index])
            axes[0, 1].set_title('Class Performance Distribution', fontweight='bold')
        
        # Model architecture summary
        if hasattr(self, 'evaluation_results') and 'model_analysis' in self.evaluation_results:
            arch_info = self.evaluation_results['model_analysis']
            
            arch_text = f"""
Model Architecture Summary:
• Total Parameters: {arch_info.get('total_parameters', 0):,}
• Model Size: {arch_info.get('model_size_mb', 0):.1f} MB
• Inference Speed: {arch_info.get('mean_inference_time_ms', 0):.1f} ms
• FPS: {arch_info.get('fps', 0):.1f}
"""
            axes[1, 0].text(0.1, 0.5, arch_text, transform=axes[1, 0].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            axes[1, 0].set_title('Model Summary', fontweight='bold')
            axes[1, 0].axis('off')
        
        # Training recommendations
        recommendations = self._generate_recommendations()
        rec_text = "Training Recommendations:\n" + "\n".join([f"• {rec}" for rec in recommendations[:5]])
        
        axes[1, 1].text(0.1, 0.5, rec_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        axes[1, 1].set_title('Recommendations', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.output_dir / "evaluation_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_per_class_chart(self):
        """Create per-class performance chart"""
        
        if not self.per_class_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        class_names = list(self.per_class_metrics.keys())
        map_scores = [info.get('mAP50_95', 0) for info in self.per_class_metrics.values()]
        difficulties = [info.get('difficulty', 'Unknown') for info in self.per_class_metrics.values()]
        
        # Color mapping for difficulty
        difficulty_colors = {
            'Easy': '#2E8B57',
            'Medium': '#4169E1', 
            'Hard': '#FFD700',
            'Very Hard': '#FF6347',
            'Unknown': '#gray'
        }
        
        colors = [difficulty_colors.get(diff, '#gray') for diff in difficulties]
        
        # Create bar chart
        bars = ax.bar(range(len(class_names)), map_scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Customize chart
        ax.set_xlabel('Traffic Light Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5:0.95', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, map_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend for difficulty
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=diff, alpha=0.7) 
                          for diff, color in difficulty_colors.items() if diff in difficulties]
        ax.legend(handles=legend_elements, title='Difficulty Level', loc='upper right')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_dir / "per_class_performance.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_difficulty_analysis(self):
        """Create difficulty vs performance analysis"""
        
        if not self.per_class_metrics:
            return
        
        # Group by difficulty
        difficulty_groups = {}
        for class_name, info in self.per_class_metrics.items():
            difficulty = info.get('difficulty', 'Unknown')
            map_score = info.get('mAP50_95', 0)
            
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(map_score)
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        difficulties = list(difficulty_groups.keys())
        scores_by_difficulty = [difficulty_groups[diff] for diff in difficulties]
        
        box_plot = ax.boxplot(scores_by_difficulty, labels=difficulties, patch_artist=True)
        
        # Color boxes
        colors = ['#2E8B57', '#4169E1', '#FFD700', '#FF6347', '#gray']
        for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5:0.95', fontsize=12, fontweight='bold')
        ax.set_title('Performance vs Difficulty Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save analysis
        analysis_path = self.output_dir / "difficulty_analysis.png"
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_speed_visualization(self):
        """Create speed benchmark visualization"""
        
        # This would create speed-related visualizations if speed data is available
        pass
    
    def _generate_recommendations(self) -> List[str]:
        """Generate training and improvement recommendations"""
        
        recommendations = []
        
        # Overall performance recommendations
        overall_map = self.validation_metrics.get('mAP50_95', 0)
        if overall_map < 0.5:
            recommendations.append("Consider training for more epochs or using a larger model")
            recommendations.append("Increase data augmentation to improve generalization")
        elif overall_map < 0.7:
            recommendations.append("Fine-tune hyperparameters for better performance")
            recommendations.append("Consider using advanced loss functions")
        
        # Class-specific recommendations
        if self.per_class_metrics:
            poor_classes = [name for name, info in self.per_class_metrics.items() 
                          if info.get('mAP50_95', 0) < 0.3]
            if poor_classes:
                recommendations.append(f"Focus on improving detection for: {', '.join(poor_classes[:3])}")
                recommendations.append("Consider class-specific data augmentation")
        
        # Balance recommendations  
        precision = self.validation_metrics.get('precision', 0)
        recall = self.validation_metrics.get('recall', 0)
        
        if precision > recall + 0.1:
            recommendations.append("Model has high precision but low recall - consider lowering confidence threshold")
        elif recall > precision + 0.1:
            recommendations.append("Model has high recall but low precision - consider increasing confidence threshold")
        
        # Default recommendations
        if not recommendations:
            recommendations.append("Model performance is good - consider testing on different datasets")
            recommendations.append("Optimize model for production deployment")
        
        return recommendations
    
    def _save_evaluation_results(self, results: Dict):
        """Save comprehensive evaluation results"""
        
        # Save JSON report
        json_path = self.output_dir / "evaluation_report.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_data = []
        if 'per_class_analysis' in results:
            for class_name, metrics in results['per_class_analysis'].items():
                csv_data.append({
                    'Class': class_name,
                    'mAP50_95': metrics.get('mAP50_95', 0),
                    'Difficulty': metrics.get('difficulty', 'Unknown'),
                    'Performance_Tier': metrics.get('performance_tier', 'Unknown')
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / "per_class_results.csv"
            df.to_csv(csv_path, index=False)
        
        # Save detailed text report
        self._create_text_report(results)
        
        self.logger.info(f"Evaluation results saved to: {self.output_dir}")
    
    def _create_text_report(self, results: Dict):
        """Create detailed text report"""
        
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAFFIC LIGHT DETECTION - MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic info
            info = results.get('evaluation_info', {})
            f.write(f"Model: {info.get('model_path', 'Unknown')}\n")
            f.write(f"Dataset: {info.get('data_config', 'Unknown')}\n")
            f.write(f"Evaluation Date: {info.get('timestamp', 'Unknown')}\n")
            f.write(f"Evaluation Time: {info.get('evaluation_time', 0):.2f} seconds\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            metrics = results.get('validation_metrics', {})
            f.write(f"mAP@0.5:        {metrics.get('mAP50', 0):.4f}\n")
            f.write(f"mAP@0.5:0.95:   {metrics.get('mAP50_95', 0):.4f}\n")
            f.write(f"Precision:      {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:         {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score:       {metrics.get('f1_score', 0):.4f}\n\n")
            
            # Per-class results
            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            class_analysis = results.get('per_class_analysis', {})
            for class_name, info in class_analysis.items():
                f.write(f"{class_name:<25}: mAP={info.get('mAP50_95', 0):.4f} "
                       f"({info.get('performance_tier', 'Unknown')})\n")
            
            f.write("\n")
            
            # Recommendations
            recommendations = self._generate_recommendations()
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

def main():
    """Main evaluation function for command line usage"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Traffic Light Detection Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='traffic_lights.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TrafficLightEvaluator(
        model_path=args.model,
        data_config=args.data,
        output_dir=args.output
    )
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📁 Results saved to: {args.output}")
    print(f"📊 Overall mAP@0.5:0.95: {results['validation_metrics']['mAP50_95']:.4f}")

if __name__ == "__main__":
    main()