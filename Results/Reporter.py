import torch

class SimpleAccReporter:
    """
    Reporter for tracking attack success metrics.
    
    Tracks:
    - Global Attack Success Rate (GASR): Overall attack success
    - Individual Attack Success Rate (IASR): Per-class attack success
    - Accuracy: Classification accuracy (1 - GASR)
    """
    
    def __init__(self):
        self.total_misclassified = 0
        self.total_samples = 0
        self.class_misclassified = {}
        self.class_total = {}
    
    def collect(self, data):
        """
        Collect statistics from a batch.
        
        Args:
            data: Tuple of (num_misclassified, batch_size) or
                  Tuple of (num_misclassified, batch_size, true_labels, pred_labels)
        """
        if len(data) == 2:
            num_misclassified, batch_size = data
            self.total_misclassified += num_misclassified.item() if torch.is_tensor(num_misclassified) else num_misclassified
            self.total_samples += batch_size
        elif len(data) == 4:
            num_misclassified, batch_size, true_labels, pred_labels = data
            self.total_misclassified += num_misclassified.item() if torch.is_tensor(num_misclassified) else num_misclassified
            self.total_samples += batch_size
            
            # Track per-class statistics
            for true_label, pred_label in zip(true_labels, pred_labels):
                true_label = true_label.item() if torch.is_tensor(true_label) else true_label
                pred_label = pred_label.item() if torch.is_tensor(pred_label) else pred_label
                
                if true_label not in self.class_total:
                    self.class_total[true_label] = 0
                    self.class_misclassified[true_label] = 0
                
                self.class_total[true_label] += 1
                if true_label != pred_label:
                    self.class_misclassified[true_label] += 1
    
    def report(self):
        """Print the attack statistics."""
        print("\n" + "="*60)
        print("ATTACK RESULTS")
        print("="*60)
        
        # Global Attack Success Rate (GASR)
        gasr = (self.total_misclassified / self.total_samples) * 100 if self.total_samples > 0 else 0
        print(f"Global Attack Success Rate (GASR): {gasr:.2f}%")
        
        # Accuracy (1 - GASR)
        accuracy = 100 - gasr
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Individual Attack Success Rate (IASR)
        if self.class_total:
            print("\n" + "-"*60)
            print("Individual Attack Success Rate (IASR) by Class:")
            print("-"*60)
            
            for class_id in sorted(self.class_total.keys()):
                total = self.class_total[class_id]
                misclassified = self.class_misclassified[class_id]
                iasr = (misclassified / total) * 100 if total > 0 else 0
                print(f"Class {class_id}: {iasr:.2f}% ({misclassified}/{total} misclassified)")
        
        print("="*60 + "\n")
    
    def reset(self):
        """Reset all statistics."""
        self.total_misclassified = 0
        self.total_samples = 0
        self.class_misclassified = {}
        self.class_total = {}