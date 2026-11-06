"""Portable entry point for the precision-assembly efficiency experiment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODES=('setup','list','freeze','preflight','train','baseline','evaluate','benchmark')


def command_for(task,arguments,*,platform=sys.platform):
    executable=ROOT/'.venv-mujoco/bin/python'
    if platform=='darwin' and task in ('baseline','evaluate','benchmark') and not any(arg in arguments for arg in ('--headless','--help','-h')):
        executable=executable.with_name('mjpython')
    if not executable.is_file():raise FileNotFoundError('Run: python -m experiments.setup')
    if task=='setup':return [str(executable),'-m','experiments.assembly_setup',*arguments]
    return [str(executable),'-m','experiments.assembly',task,*arguments]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=MODES)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    try:command=command_for(args.task,args.arguments)
    except FileNotFoundError as error:parser.exit(1,str(error)+'\n')
    raise SystemExit(subprocess.call(command,cwd=ROOT))


if __name__=='__main__':main()
