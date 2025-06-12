import numpy as np
import json
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

class AccuracyEvaluator:
    def __init__(self):
        """Initialize accuracy evaluator for test vs prediction comparison"""
        self.variable_names = [
            'Fire State', 'Temperature', 'Smoke Density', 'Visibility',
            'CO Level', 'HCN Level', 'Air Velocity', 'Thermal Radiation', 'Pressure'
        ]
        
        self.variable_units = [
            'state_enum', 'celsius', 'optical_density_per_meter', 'meters',
            'ppm', 'ppm', 'meters_per_second', 'kilowatts_per_square_meter', 'pascals'
        ]
        
        self.prediction_start = 25  # AI predictions start from timestep 25
        
    def load_data(self, json_file):
        """Load simulation data from JSON and binary files"""
        try:
            # Load metadata
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Load binary data
            bin_file = json_file.replace('.json', '.bin')
            if not os.path.exists(bin_file):
                print(f"Warning: Binary file not found: {bin_file}")
                return None, None
            
            with open(bin_file, 'rb') as f:
                data_bytes = f.read()
            
            # Reconstruct data array
            data_shape = tuple(metadata['data_shape'])
            data = np.frombuffer(data_bytes, dtype=np.float32).reshape(data_shape)
            
            return data, metadata
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return None, None
    
    def calculate_metrics(self, gt_data, pred_data, start_timestep=25):
        """Calculate accuracy metrics for all variables"""
        timesteps, height, width, variables = gt_data.shape
        
        # Only evaluate predictions after start_timestep
        if start_timestep < timesteps:
            gt_eval = gt_data[start_timestep:]
            pred_eval = pred_data[start_timestep:]
        else:
            print(f"Warning: start_timestep {start_timestep} >= total timesteps {timesteps}")
            return {}
        
        metrics = {}
        
        for var_idx in range(variables):
            var_name = self.variable_names[var_idx]
            
            # Flatten spatial dimensions for metric calculation
            gt_var = gt_eval[:, :, :, var_idx].flatten()
            pred_var = pred_eval[:, :, :, var_idx].flatten()
            
            # Calculate metrics
            mse = mean_squared_error(gt_var, pred_var)
            mae = mean_absolute_error(gt_var, pred_var)
            rmse = np.sqrt(mse)
            
            # R² score (coefficient of determination)
            try:
                r2 = r2_score(gt_var, pred_var)
            except:
                r2 = 0.0
            
            # Mean Absolute Percentage Error (MAPE)
            # Avoid division by zero
            gt_nonzero = gt_var[gt_var != 0]
            pred_nonzero = pred_var[gt_var != 0]
            
            if len(gt_nonzero) > 0:
                mape = np.mean(np.abs((gt_nonzero - pred_nonzero) / gt_nonzero)) * 100
            else:
                mape = 0.0
            
            # Accuracy (1 - normalized error)
            max_val = np.max(gt_var) if np.max(gt_var) > 0 else 1.0
            normalized_mae = mae / max_val
            accuracy = max(0, (1 - normalized_mae) * 100)
            
            metrics[var_name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
                'Accuracy': accuracy,
                'GT_Mean': np.mean(gt_var),
                'GT_Std': np.std(gt_var),
                'Pred_Mean': np.mean(pred_var),
                'Pred_Std': np.std(pred_var),
                'GT_Range': [np.min(gt_var), np.max(gt_var)],
                'Pred_Range': [np.min(pred_var), np.max(pred_var)]
            }
        
        return metrics
    
    def calculate_temporal_accuracy(self, gt_data, pred_data, start_timestep=25):
        """Calculate accuracy over time for each variable"""
        timesteps, height, width, variables = gt_data.shape
        
        if start_timestep >= timesteps:
            return {}
        
        temporal_metrics = {}
        
        for var_idx in range(variables):
            var_name = self.variable_names[var_idx]
            temporal_metrics[var_name] = {
                'timesteps': [],
                'mae': [],
                'accuracy': []
            }
            
            for t in range(start_timestep, timesteps):
                gt_frame = gt_data[t, :, :, var_idx].flatten()
                pred_frame = pred_data[t, :, :, var_idx].flatten()
                
                mae = mean_absolute_error(gt_frame, pred_frame)
                
                # Calculate accuracy
                max_val = np.max(gt_frame) if np.max(gt_frame) > 0 else 1.0
                normalized_mae = mae / max_val
                accuracy = max(0, (1 - normalized_mae) * 100)
                
                temporal_metrics[var_name]['timesteps'].append(t)
                temporal_metrics[var_name]['mae'].append(mae)
                temporal_metrics[var_name]['accuracy'].append(accuracy)
        
        return temporal_metrics
    
    def evaluate_single_file(self, test_file, pred_file):
        """Evaluate accuracy for a single test-prediction pair"""
        print(f"\nEvaluating: {os.path.basename(test_file)}")
        print("-" * 60)
        
        # Load data
        gt_data, gt_metadata = self.load_data(test_file)
        pred_data, pred_metadata = self.load_data(pred_file)
        
        if gt_data is None or pred_data is None:
            print("Error: Could not load data files")
            return None
        
        if gt_data.shape != pred_data.shape:
            print(f"Error: Shape mismatch - GT: {gt_data.shape}, Pred: {pred_data.shape}")
            return None
        
        # Calculate overall metrics
        metrics = self.calculate_metrics(gt_data, pred_data, self.prediction_start)
        
        # Calculate temporal metrics
        temporal_metrics = self.calculate_temporal_accuracy(gt_data, pred_data, self.prediction_start)
        
        # Print results
        print(f"Simulation Duration: {gt_metadata.get('simulation_duration_seconds', 0):.0f} seconds")
        print(f"Prediction Period: {self.prediction_start}s - {gt_data.shape[0]}s")
        print(f"Data Shape: {gt_data.shape}")
        print()
        
        print("ACCURACY METRICS BY VARIABLE:")
        print("=" * 80)
        print(f"{'Variable':<15} {'Accuracy':<10} {'MAE':<12} {'RMSE':<12} {'R²':<8} {'MAPE':<10}")
        print("-" * 80)
        
        overall_accuracy = 0
        for var_name, metrics_data in metrics.items():
            accuracy = metrics_data['Accuracy']
            mae = metrics_data['MAE']
            rmse = metrics_data['RMSE']
            r2 = metrics_data['R²']
            mape = metrics_data['MAPE']
            
            print(f"{var_name:<15} {accuracy:<10.1f}% {mae:<12.3f} {rmse:<12.3f} {r2:<8.3f} {mape:<10.1f}%")
            overall_accuracy += accuracy
        
        overall_accuracy /= len(metrics)
        print("-" * 80)
        print(f"{'OVERALL':<15} {overall_accuracy:<10.1f}%")
        print("=" * 80)
        
        return {
            'file': os.path.basename(test_file),
            'overall_metrics': metrics,
            'temporal_metrics': temporal_metrics,
            'overall_accuracy': overall_accuracy,
            'metadata': {
                'gt': gt_metadata,
                'pred': pred_metadata
            }
        }
    
    def evaluate_all_files(self):
        """Evaluate accuracy for all test-prediction pairs"""
        test_files = glob.glob("test_dataset/fire_simulation_*.json")
        
        if not test_files:
            print("No test files found in test_dataset/")
            return
        
        results = []
        total_accuracy = 0
        
        print("FIRE SIMULATION AI PREDICTION ACCURACY EVALUATION")
        print("=" * 80)
        
        for test_file in test_files:
            pred_file = f"pred_dataset/{os.path.basename(test_file)}"
            
            if os.path.exists(pred_file):
                result = self.evaluate_single_file(test_file, pred_file)
                if result:
                    results.append(result)
                    total_accuracy += result['overall_accuracy']
            else:
                print(f"Warning: Prediction file not found for {os.path.basename(test_file)}")
        
        if results:
            avg_accuracy = total_accuracy / len(results)
            
            print(f"\nSUMMARY:")
            print(f"Files Evaluated: {len(results)}")
            print(f"Average Overall Accuracy: {avg_accuracy:.1f}%")
            
            # Save detailed results
            self.save_results(results)
            
            # Generate plots
            self.generate_plots(results)
        else:
            print("No valid results to evaluate.")
    
    def save_results(self, results):
        """Save evaluation results to CSV and JSON"""
        # Create summary DataFrame
        summary_data = []
        for result in results:
            row = {'File': result['file'], 'Overall_Accuracy': result['overall_accuracy']}
            for var_name, metrics in result['overall_metrics'].items():
                row[f'{var_name}_Accuracy'] = metrics['Accuracy']
                row[f'{var_name}_MAE'] = metrics['MAE']
                row[f'{var_name}_R2'] = metrics['R²']
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv('accuracy_evaluation_summary.csv', index=False)
        
        # Save detailed results as JSON
        with open('accuracy_evaluation_detailed.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved:")
        print(f"  Summary: accuracy_evaluation_summary.csv")
        print(f"  Detailed: accuracy_evaluation_detailed.json")
    
    def generate_plots(self, results):
        """Generate accuracy visualization plots"""
        if not results:
            return
        
        # Create accuracy comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AI Fire Prediction Accuracy Analysis', fontsize=16)
        
        # Plot 1: Overall accuracy by variable
        var_accuracies = {}
        for var_name in self.variable_names:
            accuracies = [r['overall_metrics'][var_name]['Accuracy'] for r in results]
            var_accuracies[var_name] = np.mean(accuracies)
        
        ax1 = axes[0, 0]
        vars_short = [name.split()[0] for name in self.variable_names]
        accuracies = list(var_accuracies.values())
        bars = ax1.bar(vars_short, accuracies, color='skyblue')
        ax1.set_title('Average Accuracy by Variable')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Plot 2: MAE by variable
        ax2 = axes[0, 1]
        var_maes = {}
        for var_name in self.variable_names:
            maes = [r['overall_metrics'][var_name]['MAE'] for r in results]
            var_maes[var_name] = np.mean(maes)
        
        maes = list(var_maes.values())
        ax2.bar(vars_short, maes, color='lightcoral')
        ax2.set_title('Mean Absolute Error by Variable')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: R² score by variable
        ax3 = axes[1, 0]
        var_r2s = {}
        for var_name in self.variable_names:
            r2s = [r['overall_metrics'][var_name]['R²'] for r in results]
            var_r2s[var_name] = np.mean(r2s)
        
        r2s = list(var_r2s.values())
        bars = ax3.bar(vars_short, r2s, color='lightgreen')
        ax3.set_title('R² Score by Variable')
        ax3.set_ylabel('R² Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, r2 in zip(bars, r2s):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{r2:.3f}', ha='center', va='bottom')
        
        # Plot 4: Overall accuracy distribution
        ax4 = axes[1, 1]
        overall_accs = [r['overall_accuracy'] for r in results]
        ax4.hist(overall_accs, bins=10, color='gold', alpha=0.7, edgecolor='black')
        ax4.set_title('Overall Accuracy Distribution')
        ax4.set_xlabel('Accuracy (%)')
        ax4.set_ylabel('Frequency')
        ax4.axvline(np.mean(overall_accs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(overall_accs):.1f}%')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('accuracy_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Plots: accuracy_evaluation_plots.png")


def main():
    print("Fire Simulation AI Prediction Accuracy Evaluation")
    print("=" * 60)
    
    evaluator = AccuracyEvaluator()
    evaluator.evaluate_all_files()


if __name__ == "__main__":
    main()