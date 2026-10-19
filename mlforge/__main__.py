"""MLForge Command Line Interface entry point.

Allows direct invocation via:
    python -m mlforge <command> [options]
"""

import sys
from mlforge.cli import main

if __name__ == "__main__":
    sys.exit(main())
