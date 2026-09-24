"""A server that dies on startup, to check we quote its last words back."""

import sys

print("FATAL: missing GITHUB_TOKEN in environment", file=sys.stderr)
sys.exit(3)
