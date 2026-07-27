#!/usr/bin/env python3
"""Download CheXpert Plus PNG images from Redivis and resize them locally."""

import argparse
import os
import sys
import time
import pandas as pd
import redivis
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_and_resize(table_dir, remote_path, local_root, image_size, delete_original=True):
    """
    Download one image from Redivis, resize it, save locally.

    Skips the download entirely if the resized file already exists,
    enabling safe re-runs after an interrupted download.

    """
    local_path = os.path.join(local_root, remote_path)

    if os.path.exists(local_path):
        return remote_path, 'skipped'

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    raw_path = local_path + '.raw'

    try:
        file = table_dir.get(remote_path)
        file.download(raw_path)

        img = Image.open(raw_path).convert('RGB')
        img = img.resize((image_size, image_size))
        img.save(local_path)

        return remote_path, 'ok'
    except Exception as e:
        return remote_path, f"error: {e}"
    finally:
        if delete_original and os.path.exists(raw_path):
            os.remove(raw_path)


def run_download(paths, table_ref, local_root, image_size, max_workers, log_every=500):
    """Run the download+resize job across all given paths, with progress logging."""
    table = redivis.table(table_ref)
    table_dir = table.to_directory()

    total = len(paths)
    ok_count, skip_count, fail_count = 0, 0, 0
    failed_paths = []

    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_and_resize, table_dir, p, local_root, image_size): p
            for p in paths
        }

        for i, future in enumerate(as_completed(futures)):
            remote_path, status = future.result()

            if status == 'ok':
                ok_count += 1
            elif status == 'skipped':
                skip_count += 1
            else:
                fail_count += 1
                failed_paths.append((remote_path, status))

            if (i + 1) % log_every == 0 or (i + 1) == total:
                elapsed = time.time() - start
                print(f"progress: {i+1}/{total} — ok={ok_count} skipped={skip_count} failed={fail_count} — {elapsed:.0f}s elapsed", flush=True)

    print(f"\ndone. ok={ok_count} skipped={skip_count} failed={fail_count} out of {total}")
    if failed_paths:
        print("\nfailed paths:")
        for p, msg in failed_paths[:50]:   # cap printed failures to avoid flooding the log
            print(f"  {p}: {msg}")
        if len(failed_paths) > 50:
            print(f"  ... and {len(failed_paths) - 50} more")

    return ok_count, skip_count, fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and resize CheXpert Plus images from Redivis.")
    parser.add_argument('--parquet-path', required=True)
    parser.add_argument('--table-ref', required=True)
    parser.add_argument('--local-root', required=True)
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--max-workers', type=int, default=8)
    parser.add_argument('--log-path', default=None, help="If set, redirect stdout/stderr to this file")
    args = parser.parse_args()

    if args.log_path:
        log_file = open(args.log_path, 'w', buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file

    print(f"args: {vars(args)}")

    df = pd.read_parquet(args.parquet_path)
    if args.split != 'all':
        df = df[df['split'] == args.split]

    paths = df['path_to_image'].str.replace('train/', '', n=1, regex=False).tolist()
    print(f"downloading {len(paths)} images for split={args.split}")

    run_download(paths, args.table_ref, args.local_root, args.image_size, args.max_workers)