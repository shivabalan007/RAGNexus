import sys
from pathlib import Path
from getpass import getpass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# Load project environment variables
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from ragnexus.core.security import hash_password, verify_password

password = getpass("Enter new password: ")
confirm = getpass("Confirm new password: ")

if password != confirm:
    raise ValueError("Passwords do not match")

new_hash = hash_password(password)

if not verify_password(password, new_hash):
    raise RuntimeError("Hash verification failed")

print("\nGenerated Argon2 hash:")
print(new_hash)