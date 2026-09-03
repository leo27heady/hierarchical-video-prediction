"""One-command CPU-only editorial assembly, tests and optional typeset preview."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from refactor_dissertation_v2 import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preview',action='store_true');p.add_argument('--browser-check',action='store_true');a=p.parse_args()
    for script in ['make_dissertation_v2_assets.py','refactor_dissertation_v2.py','register_dissertation_v2_evidence.py','register_dissertation_scientific_dependencies.py','audit_dissertation_v2.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_dissertation_refactor.py','-v'],cwd=ROOT,check=True)
    if a.preview or a.browser_check:
        node=os.environ.get('DISSERTATION_NODE',str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'))
        subprocess.run([node,'scripts/build_dissertation_v2_preview.mjs'],cwd=ROOT,check=True)
        if a.browser_check:subprocess.run([node,'scripts/check_dissertation_v2_preview.mjs'],cwd=ROOT,check=True)


if __name__=='__main__':main()
