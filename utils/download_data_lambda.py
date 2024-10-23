"""
Script to download LeanDojo Benchmark and LeanDojo Benchmark 4 into `./data`.

In some filesystems the original download_data.py will raise error due to query params. Use this to debug.
"""

import os
import argparse
from hashlib import md5
from loguru import logger


LEANDOJO_BENCHMARK_4_URL = "https://zenodo.org/records/10929138/files/leandojo_benchmark_4.tar.gz"
DOWNLOADS = {
    LEANDOJO_BENCHMARK_4_URL: "84a75ce552b31731165d55542b1aaca9",
}


def check_md5(filename: str, gt_hashcode: str) -> bool:
    """
    Check the MD5 of a file against the ground truth.
    """
    if not os.path.exists(filename):
        return False
    # The file could be large.
    # See https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file.
    inp = open(filename, "rb")
    hasher = md5()
    while True:
        block = inp.read(64 * (1 << 20))
        if not block:
            break
        hasher.update(block)
    calculated_hash = hasher.hexdigest()
    logger.info(f"Calculated MD5 hash: {calculated_hash}")
    logger.info(f"Expected MD5 hash: {gt_hashcode}")
    return calculated_hash == gt_hashcode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()
    logger.info(args)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    for url, hashcode in DOWNLOADS.items():
        logger.info(f"Downloading {url}")
        path = f"{args.data_path}/{os.path.basename(url.split('?')[0])}"
        logger.info(f"File path: {path}")
        logger.info(f"wget command: wget '{url}' -O {path}")
        os.system(f"wget '{url}' -O {path}")
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
        else:
            logger.info(f"File downloaded: {path}")
        if not check_md5(path, hashcode):
            raise RuntimeError(f"MD5 of {path} does not match the ground truth.")

        logger.info(f"Extracting {path}")
        os.system(f"tar -xf {path} -C {args.data_path}")

        logger.info(f"Removing {path}")
        os.remove(path)

    logger.info("Done!")


if __name__ == "__main__":
    main()