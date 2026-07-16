"""
TO RESOLVE: 
X:/imgAI/ComfyUI_Intel/python_embeded/python.exe -m pip install PACKAGE_NAME_HERE
X:/imgAI/ComfyUI_Intel/python_embeded/python.exe -c "import PACKAGE_NAME_HERE; print('huzzah!')"

"""
import ast
import os
import sys
import importlib

def get_imports_from_file(file_path):
    """Parses a Python file and extracts all top-level module names being imported."""
    modules = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), file_path)
        
        for node in ast.walk(tree):
            # Handles: import foo, bar
            if isinstance(node, ast.Import):
                for name in node.names:
                    modules.add(name.name.split('.')[0])
            # Handles: from foo import bar
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:  # level 0 means absolute import, not relative
                    modules.add(node.module.split('.')[0])
    except Exception as e:
        print(f"⚠️ Could not parse {file_path}: {e}")
    return modules

def scan_project_imports(root_dir):
    """Scans all nested folders for .py files, gathers imports, and validates them."""
    all_imports = set()
    
    # 1. Gather all imported modules across the directory tree
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py') and filename != os.path.basename(__file__):
                file_path = os.path.join(dirpath, filename)
                all_imports.update(get_imports_from_file(file_path))
    
    # 2. Filter out standard built-in modules and local project files
    # We do this by checking if the module name matches a folder inside our project
    project_folders = {d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))}
    external_imports = all_imports - project_folders - set(sys.builtin_module_names)

    print(f"🔍 Found {len(external_imports)} unique external modules to check...\n")
    
    # 3. Test the remaining external modules
    failed_imports = []
    for module in sorted(external_imports):
        try:
            importlib.import_module(module)
            print(f"✅ {module:20} -> SUCCESS")
        except ImportError:
            print(f"❌ {module:20} -> MISSING / FAILED")
            failed_imports.append(module)
            
    # Final Report
    print("\n" + "="*40)
    if failed_imports:
        print(f"🚨 Validation Failed! Missing packages: {', '.join(failed_imports)}")
        print("Run: pip install " + " ".join(failed_imports))
    else:
        print("🎉 Huzzah! All imports verified successfully across the project.")
    print("="*40)

if __name__ == "__main__":
    # Scans the current folder where the script is located
    current_directory = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    scan_project_imports(current_directory)