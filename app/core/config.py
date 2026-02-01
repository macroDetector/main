import os
import sys

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def update_env(updates: dict):
    env_path = ".env"
    env_dict = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    env_dict[key] = val
    
    env_dict.update(updates)
    with open(env_path, "w", encoding="utf-8") as f:
        for key, val in env_dict.items():
            f.write(f"{key}={val}\n")