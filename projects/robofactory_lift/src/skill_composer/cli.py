"""The multi-task cell is the default; the committed v1 remains a baseline."""
import sys


def main():
    if len(sys.argv)>1 and sys.argv[1]=='compose':
        sys.argv.pop(1)
        from .compose.cli import main as run
    elif len(sys.argv)>1 and sys.argv[1]=='baseline':
        sys.argv.pop(1)
        from .baseline_cli import main as run
    else:
        from .cell.cli import main as run
    run()
