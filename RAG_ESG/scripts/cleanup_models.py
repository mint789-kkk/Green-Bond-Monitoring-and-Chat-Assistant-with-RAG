"""Remove cached/unused models."""

import shutil
from pathlib import Path


def main():
    target = Path.home() / ".deskrag" / "models"
    if target.exists():
        shutil.rmtree(target)
        print(f"Removed {target}")
    else:
        print("No models directory found.")


if __name__ == "__main__":
    main()



