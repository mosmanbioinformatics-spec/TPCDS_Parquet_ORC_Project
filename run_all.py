import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    command = [sys.executable, '-u', str(ROOT / 'src' / script), *map(str, args)]
    print('\n' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-only', action='store_true')
    parser.add_argument('--check-content', action='store_true')
    parser.add_argument('--keep-data', action='store_true')
    args = parser.parse_args()
    if not args.analysis_only:
        extra = ['--check-content'] if args.check_content else []
        for scale in (1, 2, 5):
            run('prepare_data.py', '--scale', scale)
            run('experiment.py', '--scale', scale, *extra)
            if scale == 1:
                run('data_types.py', '--scale', scale, *extra)
                run('sensitivity.py', '--scale', scale, *extra)
            if not args.keep_data:
                shutil.rmtree(ROOT / 'work/data' / f'tpcds_sf{scale}')
    run('statistical_analysis.py')
    print('\nResults: results/ | Figures: figures/', flush=True)


if __name__ == '__main__':
    main()
