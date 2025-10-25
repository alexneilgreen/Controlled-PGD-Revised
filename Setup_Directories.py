import os

def setup_project_structure():
    """
    Create the necessary directory structure for the CPGD project.
    """
    directories = [
        'Architecture',
        'Attacks',
        'Data',
        'Models',
        'Results'
    ]
    
    print("Setting up project directory structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}/")
    
    # Create __init__.py files for packages
    init_files = [
        'Architecture/__init__.py',
        'Attacks/__init__.py',
        'Data/__init__.py',
        'Models/__init__.py',
        'Results/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass
            print(f"✓ Created: {init_file}")
    
    print("\nProject structure setup complete!")
    print("\nYour directory structure should now look like:")
    print("""
    project/
    ├── Architecture/
    │   ├── __init__.py
    │   ├── ResNet.py
    │   └── ViT.py
    ├── Attacks/
    │   ├── __init__.py
    │   ├── Classes.py
    │   ├── PGD.py
    │   └── CPGD.py
    ├── Data/            (for datasets)
    │   ├── __init__.py
    ├── Models/          (for saved trained models)
    │   ├── __init__.py
    ├── Results/
    │   ├── __init__.py
    │   └── reporter.py
    ├── Data_Loader.py
    └── Main.py
    """)

if __name__ == "__main__":
    setup_project_structure()