#!/usr/bin/env python3
"""
Simple parallel config runner with logging
"""
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

def run_config(config_path):
    """Run train.py with a single config and save logs"""
    # Create log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("logs") / f"{config_path.stem}_{timestamp}.txt"
    log_file.parent.mkdir(exist_ok=True)
    
    cmd = ["python", "train.py", str(config_path), "--debug"]
    
    try:
        # Run and capture output
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("-" * 80 + "\n\n")
            
            result = subprocess.run(
                cmd, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                text=True
            )
        
        return config_path.name, result.returncode == 0, log_file.name
    except Exception as e:
        return config_path.name, False, f"error: {str(e)}"

def main():
    # Get config directory
    config_dir = Path("../configs")
    configs = list(config_dir.glob("*.yaml"))
    
    if not configs:
        print("No configs found!")
        return
    
    print(f"Found {len(configs)} configs")
    print(f"Logs will be saved to logs/")
    print("-" * 40)
    
    # Track status
    completed_configs = []
    running_configs = {cfg.name: cfg for cfg in configs}
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_config, cfg): cfg for cfg in configs}
        
        # Show initial status
        print("\n🚀 STATUS:")
        print(f"⏳ Running: {', '.join(running_configs.keys())}")
        print(f"✅ Completed: None yet")
        print("-" * 40)
        
        for future in as_completed(futures):
            config_name, success, log_name = future.result()
            status = "✓" if success else "✗"
            
            # Update tracking
            completed_configs.append(f"{status} {config_name}")
            del running_configs[config_name]
            
            # Clear previous lines and show updated status
            print(f"\n🚀 STATUS UPDATE:")
            print(f"⏳ Running ({len(running_configs)}): {', '.join(running_configs.keys()) if running_configs else 'None'}")
            print(f"✅ Completed ({len(completed_configs)}): {', '.join(completed_configs)}")
            print(f"📄 Latest: {status} {config_name} → {log_name}")
            print("-" * 40)
    
    # Final summary
    print("\n🎉 ALL DONE!")
    successful = sum(1 for cfg in completed_configs if cfg.startswith("✓"))
    print(f"Success: {successful}/{len(configs)}")

if __name__ == "__main__":
    main()