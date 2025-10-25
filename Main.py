import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

# Import custom modules
from Data_Loader import get_dataloader, get_available_datasets
from Architecture.ResNet import ResNet18
from Architecture.ViT import VisionTransformer
from Attacks.Classes import UntargetedAttack, TargetedAttack

def get_num_classes(dataset_name):
    """Get number of classes for each dataset."""
    if dataset_name.lower() == 'cifar100':
        return 100
    else:  # mnist, cifar10, stl10
        return 10

def get_model(model_name, num_classes, in_channels=3):
    """Initialize model based on name."""
    if model_name.lower() == 'resnet':
        return ResNet18(num_classes=num_classes, in_channels=in_channels)
    elif model_name.lower() == 'vit':
        return VisionTransformer(
            img_size=96,
            patch_size=8,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=384,
            depth=6,
            n_heads=6,
            mlp_ratio=4.0,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, train_loader, test_loader, epochs, lr, device, save_path):
    """Train a model and save it."""
    print(f"\nTraining model for {epochs} epochs with lr={lr}...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.to(device)
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
        
        train_acc = 100. * correct / total
        scheduler.step()
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%\n')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
            print(f'Model saved with accuracy: {test_acc:.2f}%')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')
    return model

def get_available_models():
    """Get list of available trained models."""
    models_dir = './Models'
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return model_files

def load_trained_model(model_path, model_name, num_classes, in_channels, device):
    """Load a trained model from file."""
    model = get_model(model_name, num_classes, in_channels)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def get_class_mapping(num_classes):
    """Prompt user for class mapping for CPGD."""
    print("\nPlease input Matrix Mapping Values")
    mapping = {}
    for i in range(num_classes):
        while True:
            try:
                target = int(input(f"Class {i} -> "))
                if 0 <= target < num_classes:
                    mapping[i] = target
                    break
                else:
                    print(f"Invalid target. Must be between 0 and {num_classes-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return mapping

def train_models_mode(args):
    """Handle training models mode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create Models directory if it doesn't exist
    os.makedirs('./Models', exist_ok=True)
    
    # Determine which models and datasets to train
    if args.model == 'all':
        models_to_train = ['resnet', 'vit']
    else:
        models_to_train = [args.model]
    
    if args.dataset == 'all':
        datasets_to_train = get_available_datasets()
    else:
        datasets_to_train = [args.dataset]
    
    # Train all combinations
    for model_name in models_to_train:
        for dataset_name in datasets_to_train:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()} on {dataset_name.upper()}")
            print(f"{'='*60}")
            
            # Check if model already exists
            save_name = f"{model_name}_{dataset_name}.pth"
            save_path = os.path.join('./Models', save_name)
            
            if os.path.exists(save_path) and not args.retrain:
                print(f"Model {save_name} already exists. Skipping...")
                print("Use --retrain flag to retrain existing models.")
                continue
            
            # Get number of classes and input channels
            num_classes = get_num_classes(dataset_name)
            in_channels = 1 if dataset_name.lower() == 'mnist' else 3
            
            # Get dataloaders
            train_loader = get_dataloader(
                dataset_name=dataset_name,
                split='train',
                batch_size=args.batch_size,
                shuffle=True
            )
            
            test_loader = get_dataloader(
                dataset_name=dataset_name,
                split='test',
                batch_size=args.batch_size,
                shuffle=False
            )
            
            # Initialize model
            model = get_model(model_name, num_classes, in_channels)
            
            # Train model
            train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                save_path=save_path
            )

def attack_models_mode(args):
    """Handle attacking models mode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        print("No trained models found. Please train models first.")
        return
    
    print("\nAvailable trained models:")
    for idx, model_file in enumerate(available_models):
        print(f"{idx+1}. {model_file}")
    
    # Select model
    while True:
        try:
            selection = int(input("\nSelect model number: ")) - 1
            if 0 <= selection < len(available_models):
                selected_model_file = available_models[selection]
                break
            else:
                print(f"Invalid selection. Please choose 1-{len(available_models)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Parse model information from filename
    model_parts = selected_model_file.replace('.pth', '').split('_')
    model_name = model_parts[0]
    dataset_name = model_parts[1]
    
    print(f"\nSelected: {model_name.upper()} trained on {dataset_name.upper()}")
    
    # Get dataset info
    num_classes = get_num_classes(dataset_name)
    in_channels = 1 if dataset_name.lower() == 'mnist' else 3
    
    # Load model
    model_path = os.path.join('./Models', selected_model_file)
    model = load_trained_model(model_path, model_name, num_classes, in_channels, device)
    print("Model loaded successfully!")
    
    # Get test dataloader
    test_loader = get_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Select attack type
    print("\nSelect attack type:")
    print("1. PGD (Untargeted)")
    print("2. CPGD (Targeted)")
    
    while True:
        try:
            attack_choice = int(input("\nSelect attack (1 or 2): "))
            if attack_choice in [1, 2]:
                break
            else:
                print("Invalid selection. Please choose 1 or 2")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Setup loss function
    loss_fn = nn.CrossEntropyLoss()
    
    if attack_choice == 1:
        # PGD Attack
        print("\nExecuting PGD (Untargeted) Attack...")
        attack = UntargetedAttack(
            model=model,
            loss=loss_fn,
            dataloader=test_loader,
            iterations=args.iterations,
            tolerance=args.tolerance,
            lr=args.attack_lr
        )
        attack.execute_attack()
    else:
        # CPGD Attack
        print("\nExecuting CPGD (Targeted) Attack...")
        mapping = get_class_mapping(num_classes)
        print(f"\nClass Mapping Saved")
        
        attack = TargetedAttack(
            model=model,
            loss=loss_fn,
            dataloader=test_loader,
            num_classes=num_classes,
            iterations=args.iterations,
            tolerance=args.tolerance,
            lr=args.attack_lr
        )
        attack.execute_attack()

def main():
    parser = argparse.ArgumentParser(description='Controlled PGD Project')
    parser.add_argument('--mode', type=str, choices=['train', 'attack'], required=True,
                       help='Mode: train models or implement attacks')
    
    # Training arguments
    parser.add_argument('--model', type=str, choices=['resnet', 'vit', 'all'], default='all',
                       help='Model architecture to train')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'all'], 
                       default='all', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--retrain', action='store_true', help='Retrain existing models')
    
    # Attack arguments
    parser.add_argument('--iterations', type=int, default=100, help='Number of attack iterations')
    parser.add_argument('--tolerance', type=float, default=0.000001, help='Attack convergence tolerance')
    parser.add_argument('--attack_lr', type=float, default=0.01, help='Attack learning rate')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_models_mode(args)
    elif args.mode == 'attack':
        attack_models_mode(args)

if __name__ == "__main__":
    main()