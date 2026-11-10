from getpass import getpass
from pwdlib import PasswordHash

# Same configuration as your security.py
_password_hash = PasswordHash.recommended()

password = getpass("Enter new password: ")
confirm = getpass("Confirm new password: ")

if password != confirm:
    raise ValueError("Passwords do not match")

# Generate Argon2 hash
new_hash = _password_hash.hash(password)

# Verify before updating database
if not _password_hash.verify(password, new_hash):
    raise RuntimeError("Password verification failed")

print("\nHash generated and verified successfully:")
print(new_hash)