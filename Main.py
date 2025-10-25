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

def get_image_size_for_model(model_name, dataset_name):
    """Get appropriate image size based on model architecture."""
    if model_name.lower() == 'vit':
        return 224  # ViT uses 224x224
    else:  # ResNet
        # Use original dataset sizes
        if dataset_name.lower() in ['mnist', 'cifar10', 'cifar100']:
            return 32
        else:  # stl10
            return 96

def get_model(model_name, num_classes, in_channels=3, img_size=32):
    """Initialize model based on name."""
    if model_name.lower() == 'resnet':
        return ResNet18(num_classes=num_classes, in_channels=in_channels)
    elif model_name.lower() == 'vit':
        # ViT configuration for 224x224 images
        return VisionTransformer(
            img_size=img_size,
            patch_size=16,  # 224/16 = 14 patches per side
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.0,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, train_loader, test_loader, epochs, lr, device, save_path):
    """Train a model with mixed precision and save it."""
    print(f"\nTraining model for {epochs} epochs with lr={lr}...")
    print(f"Using mixed precision training (FP16)...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    
    model.to(device)
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%\n')
        
        # Save best model and check early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
            print(f'Model saved with accuracy: {test_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'No improvement for {patience} consecutive epochs')
                break
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')
    return model

def get_available_models():
    """Get list of available trained models."""
    models_dir = './Models'
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return model_files

def load_trained_model(model_path, model_name, num_classes, in_channels, device, img_size):
    """Load a trained model from file."""
    model = get_model(model_name, num_classes, in_channels, img_size)
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
    
    os.makedirs('./Models', exist_ok=True)
    
    if args.model == 'all':
        models_to_train = ['resnet', 'vit']
    else:
        models_to_train = [args.model]
    
    if args.dataset == 'all':
        datasets_to_train = get_available_datasets()
    else:
        datasets_to_train = [args.dataset]
    
    for model_name in models_to_train:
        for dataset_name in datasets_to_train:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()} on {dataset_name.upper()}")
            print(f"{'='*60}")
            
            save_name = f"{model_name}_{dataset_name}.pth"
            save_path = os.path.join('./Models', save_name)
            
            if os.path.exists(save_path) and not args.retrain:
                print(f"Model {save_name} already exists. Skipping...")
                print("Use --retrain flag to retrain existing models.")
                continue
            
            num_classes = get_num_classes(dataset_name)
            in_channels = 3  # All datasets are 3 channels
            img_size = get_image_size_for_model(model_name, dataset_name)
            
            print(f"Image size for {model_name}: {img_size}x{img_size}")
            
            train_loader = get_dataloader(
                dataset_name=dataset_name,
                split='train',
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                target_size=img_size
            )
            
            test_loader = get_dataloader(
                dataset_name=dataset_name,
                split='test',
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                target_size=img_size
            )
            
            model = get_model(model_name, num_classes, in_channels, img_size)
            
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
    
    available_models = get_available_models()
    
    if not available_models:
        print("No trained models found. Please train models first.")
        return
    
    print("\nAvailable trained models:")
    for idx, model_file in enumerate(available_models):
        print(f"{idx+1}. {model_file}")
    
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
    
    model_parts = selected_model_file.replace('.pth', '').split('_')
    model_name = model_parts[0]
    dataset_name = model_parts[1]
    
    print(f"\nSelected: {model_name.upper()} trained on {dataset_name.upper()}")
    
    num_classes = get_num_classes(dataset_name)
    in_channels = 3
    img_size = get_image_size_for_model(model_name, dataset_name)
    
    model_path = os.path.join('./Models', selected_model_file)
    model = load_trained_model(model_path, model_name, num_classes, in_channels, device, img_size)
    print("Model loaded successfully!")
    
    test_loader = get_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        target_size=img_size
    )
    
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
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Create Results directory
    os.makedirs('./Results', exist_ok=True)
    
    if attack_choice == 1:
        print("\nExecuting PGD (Untargeted) Attack...")
        save_path = f"./Results/{model_name}_{dataset_name}_pgd.txt"
        attack = UntargetedAttack(
            model=model,
            loss=loss_fn,
            dataloader=test_loader,
            save_path=save_path,
            iterations=args.iterations,
            tolerance=args.tolerance,
            lr=args.attack_lr
        )
        attack.execute_attack()
    else:
        print("\nExecuting CPGD (Targeted) Attack...")
        mapping = get_class_mapping(num_classes)
        print(f"\nClass Mapping: {mapping}")
        
        save_path = f"./Results/{model_name}_{dataset_name}_cpgd.txt"
        attack = TargetedAttack(
            model=model,
            loss=loss_fn,
            dataloader=test_loader,
            num_classes=num_classes,
            mapping=mapping,
            save_path=save_path,
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
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
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