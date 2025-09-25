import os
import re
import glob
import argparse
import subprocess

def parse_filename(fname):
    # fname: basename of file
    # Returns dict with parsed args or None if essential parts missing
    out = {}
    # data is first token before underscore
    tokens = fname.split('_')
    out['data'] = tokens[0] if tokens else None

    # loss: find "loss" followed by alnum/_ chars
    m = re.search(r'loss([A-Za-z0-9_]+)', fname)
    out['loss'] = m.group(1) if m else None

    # seed: final numeric token before _models.pkl
    m = re.search(r'_(\d+)_models\.pkl$', fname)
    out['seed'] = m.group(1) if m else None

    # nl and hs
    m = re.search(r'_nl(\d+)', fname)
    out['nl'] = m.group(1) if m else None
    m = re.search(r'_hs(\d+)', fname)
    out['hs'] = m.group(1) if m else None

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='elevator', help='dataset directory to search under results/lump')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to pass to main_inference.py')
    parser.add_argument('--scratch', type=str, default=os.environ.get('SCRATCH'), help='SCRATCH base dir (env SCRATCH used if not provided)')
    parser.add_argument('--main', type=str, default=os.path.join(os.path.dirname(__file__), 'main_inference.py'), help='path to main_inference.py')
    parser.add_argument('--dry-run', action='store_true', help='only print commands')
    args = parser.parse_args()

    if not args.scratch:
        raise RuntimeError('SCRATCH not set and --scratch not provided')

    base = os.path.join(args.scratch, 'results', 'lump', args.dataset)
    pattern = os.path.join(base, '*', '*models.pkl')
    candidates = glob.glob(pattern)
    if not candidates:
        print(f'No candidates found with pattern: {pattern}')
        return

    for fp in sorted(candidates):
        fname = os.path.basename(fp)
        hp_dir = os.path.basename(os.path.dirname(fp))

        # filter: hp_dir must be present in filename (hyperparam dir consistency)
        if hp_dir.replace('-', '') not in fname:
            # skip mismatched hp directories
            print(f'Skipping mismatched hp directory: {fp}')
            continue
            
        if args.dataset not in fname:
            # skip files not matching dataset
            print(f'Skipping file not matching dataset: {fname}')
            continue

        # skip if output (without "_models") already exists
        out_fp = fp.replace('_models.pkl', '.pkl')
        if os.path.exists(out_fp):
            # already processed
            print(f'Skipping existing output: {out_fp}')
            continue

        parsed = parse_filename(fname)
        if not parsed['data'] or not parsed['loss'] or not parsed['seed']:
            # skip files that don't have minimal parseable info
            print(f'Skipping unparseable file: {fname}')
            continue

        cmd = [
            'python', args.main,
            '--models_path', fp,
            '--data', parsed['data'],
            '--loss', parsed['loss'],
            '--seed', parsed['seed'],
            '--residual', '1',  # residual forced to 1 per request
            '--gpu', str(args.gpu)
        ]
        if parsed.get('nl'):
            cmd += ['--nl', parsed['nl']]
        if parsed.get('hs'):
            cmd += ['--hs', parsed['hs']]

        cmd_str = ' '.join(map(lambda s: f'"{s}"' if ' ' in s else s, cmd))
        if args.dry_run:
            print(cmd_str)
        else:
            print(f'Executing: {cmd_str}')
            subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()