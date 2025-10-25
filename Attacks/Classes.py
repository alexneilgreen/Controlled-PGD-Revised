from Attacks.PGD import PGD
from Attacks.CPGD import CPGD
from Results.Reporter import SimpleAccReporter
from torch import device, cuda
import torch

dev = device("cuda" if cuda.is_available() else "cpu")

class UntargetedAttack:
    def __init__(self, model, loss, dataloader, **kwargs):
        self.model = model
        self.loss = loss
        self.dataloader = dataloader

        iterations = kwargs.get('iterations', 100)
        tolerance = kwargs.get('tolerance', 0.000001)
        epsilon = kwargs.get('epsilon', 0.3)
        alpha = kwargs.get('alpha', 0.01)
        
        # Override with lr if provided (for backward compatibility)
        if 'lr' in kwargs:
            alpha = kwargs['lr']

        self.pgd = PGD(iterations=iterations, tolerance=tolerance, 
                      epsilon=epsilon, alpha=alpha)
        self.reporter = SimpleAccReporter()

    def execute_attack(self):
        print("\nExecuting PGD Attack...")
        print(f"Processing {len(self.dataloader)} batches...")
        
        for batch_idx, (data, label) in enumerate(self.dataloader):
            data = data.to(device=dev)
            label = label.to(device=dev)
            
            # Generate adversarial examples
            advx = self.pgd(data, label, self.pgd.alpha, self.model, self.loss)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(advx)
                _, advlabel = outputs.max(1)
            
            # Collect statistics with class information
            misclassified = (advlabel != label)
            self.reporter.collect((
                misclassified.sum(),
                advx.size(dim=0),
                label.cpu(),
                advlabel.cpu()
            ))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(self.dataloader)} batches")

        self.reporter.report()

class TargetedAttack:
    def __init__(self, model, loss, dataloader, num_classes=10, mapping=None, **kwargs):
        self.model = model
        self.loss = loss
        self.dataloader = dataloader
        self.num_classes = num_classes

        iterations = kwargs.get('iterations', 100)
        tolerance = kwargs.get('tolerance', 0.000001)
        epsilon = kwargs.get('epsilon', 0.3)
        alpha = kwargs.get('alpha', 0.01)
        
        # Override with lr if provided (for backward compatibility)
        if 'lr' in kwargs:
            alpha = kwargs['lr']

        self.cpgd = CPGD(iterations=iterations, tolerance=tolerance, 
                        num_classes=num_classes, epsilon=epsilon, alpha=alpha)
        
        # Set mapping if provided
        if mapping is not None:
            self.cpgd.set_mapping(mapping)
        
        self.reporter = SimpleAccReporter()
        self.targeted_reporter = TargetedSuccessReporter(num_classes, mapping)

    def execute_attack(self):
        print("\nExecuting CPGD Attack...")
        print(f"Processing {len(self.dataloader)} batches...")
        
        for batch_idx, (data, label) in enumerate(self.dataloader):
            data = data.to(device=dev)
            label = label.to(device=dev)
            
            # Generate adversarial examples
            advx = self.cpgd(data, label, self.cpgd.alpha, self.model, self.loss)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(advx)
                _, advlabel = outputs.max(1)
            
            # Get target labels for this batch
            target_labels = self.cpgd.get_target_labels(label)
            
            # Collect general statistics
            misclassified = (advlabel != label)
            self.reporter.collect((
                misclassified.sum(),
                advx.size(dim=0),
                label.cpu(),
                advlabel.cpu()
            ))
            
            # Collect targeted attack statistics
            self.targeted_reporter.collect(
                label.cpu(),
                advlabel.cpu(),
                target_labels.cpu()
            )
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(self.dataloader)} batches")

        self.reporter.report()
        self.targeted_reporter.report()


class TargetedSuccessReporter:
    """
    Reporter specifically for targeted attacks (CPGD).
    Tracks how well the attack achieved the targeted misclassifications.
    """
    
    def __init__(self, num_classes, mapping):
        self.num_classes = num_classes
        self.mapping = mapping
        self.targeted_success = {}  # source_class -> {target: count, total: count}
        
        # Initialize tracking for each source class
        for source in range(num_classes):
            self.targeted_success[source] = {
                'achieved_target': 0,
                'total': 0
            }
    
    def collect(self, true_labels, pred_labels, target_labels):
        """
        Collect targeted attack statistics.
        
        Args:
            true_labels: Original true labels
            pred_labels: Predicted labels after attack
            target_labels: Intended target labels from mapping
        """
        for true_label, pred_label, target_label in zip(true_labels, pred_labels, target_labels):
            true_label = true_label.item() if torch.is_tensor(true_label) else true_label
            pred_label = pred_label.item() if torch.is_tensor(pred_label) else pred_label
            target_label = target_label.item() if torch.is_tensor(target_label) else target_label
            
            self.targeted_success[true_label]['total'] += 1
            
            # Check if attack achieved the target
            if pred_label == target_label:
                self.targeted_success[true_label]['achieved_target'] += 1
    
    def report(self):
        """Print targeted attack statistics."""
        print("\n" + "="*60)
        print("TARGETED ATTACK SPECIFICS (CPGD)")
        print("="*60)
        
        total_targeted_success = 0
        total_samples = 0
        
        print("\nTargeted Success Rate by Class:")
        print("-"*60)
        print(f"{'Class':<8} {'Target':<8} {'Success Rate':<15} {'Samples'}")
        print("-"*60)
        
        for source_class in sorted(self.targeted_success.keys()):
            stats = self.targeted_success[source_class]
            total = stats['total']
            achieved = stats['achieved_target']
            
            if total > 0:
                success_rate = (achieved / total) * 100
                target_class = self.mapping[source_class]
                print(f"{source_class:<8} {target_class:<8} {success_rate:>6.2f}%         {achieved}/{total}")
                
                total_targeted_success += achieved
                total_samples += total
        
        print("-"*60)
        
        if total_samples > 0:
            overall_targeted_success = (total_targeted_success / total_samples) * 100
            print(f"\nOverall Targeted Success Rate: {overall_targeted_success:.2f}%")
            print(f"(Percentage of samples that were misclassified to the intended target)")
        
        print("="*60 + "\n")