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
    splitted = fname.split('loss')[1]
    loss = splitted.split('_ens1')[0] 
    out['loss'] = loss

    # seed: final numeric token before _models or _alpha
    # m = re.search(r'_(\d+)_models\.pkl$', fname)
    m = re.search(r'_(\d+)_(models|alpha)', fname)
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
    parser.add_argument('--base', type=str, help='base directory under which to search for models')
    parser.add_argument('--main', type=str, default=os.path.join(os.path.dirname(__file__), 'main_inference.py'), help='path to main_inference.py')
    parser.add_argument('--dry-run', action='store_true', help='only print commands')
    args = parser.parse_args()

    base = args.base
    pattern = os.path.join(base, args.dataset, '*', '*models.pkl')
    candidates = glob.glob(pattern)
    if not candidates:
        print(f'No candidates found with pattern: {pattern}')
        return

    total_unskipped = 0
    for fp in sorted(candidates):
        fname = os.path.basename(fp)
        hp_dir = os.path.basename(os.path.dirname(fp))

        if (hp_dir != 'default' and hp_dir.replace('-', '') not in fname) or (hp_dir == 'default' and 'nl2_hs64' not in fname):
            print(f'Skipping mismatched hp directory: {fp}')
            continue

        parsed = parse_filename(fname)
        if not parsed['data'] or not parsed['loss'] or not parsed['seed']:
            # skip files that don't have minimal parseable info
            print(f'Skipping unparseable file: {fname}')
            continue

        # Ensure the parsed data matches the requested dataset
        if parsed['data'] != args.dataset:
            print(f"Skipping file whose parsed data '{parsed['data']}' != dataset '{args.dataset}': {fname}")
            continue

        # skip if output (without "_models") already exists
        out_fp = fp.replace('_models.pkl', '.pkl')
        if os.path.exists(out_fp):
            # already processed
            print(f'Skipping existing output: {out_fp}')
            continue

        # set save_dir to the directory where the model file lives
        save_dir = os.path.dirname(fp)

        cmd = [
            'python', args.main,
            '--models_path', fp,
            '--save_dir', save_dir,
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
            total_unskipped += 1
        else:
            print(f'Executing: {cmd_str}')
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")
                print(f"Command: {e.cmd}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
    print(f'Total unskipped files processed or to be processed: {total_unskipped}')

if __name__ == '__main__':
    main()