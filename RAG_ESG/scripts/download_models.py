"""Download required models (placeholder)."""

from pathlib import Path


def main():
    target = Path.home() / ".deskrag" / "models"
    target.mkdir(parents=True, exist_ok=True)
    print(f"Models directory prepared at {target}")
    print("Add huggingface-cli download commands here for CLIP and LLM weights.")


if __name__ == "__main__":
    main()



