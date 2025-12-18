"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk
- stream files online from Hugging Face without downloading

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# 尝试导入 datasets 库用于流式加载
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# 尝试导入 huggingface_hub 以支持更好的重连配置
try:
    from huggingface_hub import configure_http_session
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
DATASET_NAME = "karpathy/fineweb-edu-100b-shuffle"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# 在线流式加载的配置参数
STREAMING_CONFIG = {
    "timeout": 60,  # 单个请求超时时间（秒）
    "max_retries": 5,  # 最大重试次数
    "retry_delay": 2,  # 重试延迟（秒）
}

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# 在线流式加载函数
def configure_streaming_session():
    """
    配置 Hugging Face 的 HTTP 会话，增加超时和重试时间。
    """
    if not HAS_HF_HUB:
        print("Warning: huggingface_hub not installed, using default session settings")
        return

    try:
        import requests
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry

        session = requests.Session()

        # 配置重试策略
        retry_strategy = Retry(
            total=STREAMING_CONFIG["max_retries"],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        configure_http_session(session)
        print(f"✓ Configured streaming session: timeout={STREAMING_CONFIG['timeout']}s, max_retries={STREAMING_CONFIG['max_retries']}")
    except Exception as e:
        print(f"Warning: Failed to configure streaming session: {e}")

def parquets_iter_batched_streaming(split, start=0, step=1, cache_examples=False):
    """
    Online streaming iteration through Hugging Face dataset without downloading.

    Args:
        split: "train" or "val" (online dataset only has "train", so "val" uses train split)
        start: starting index for DDP
        step: step size for DDP
        cache_examples: whether to cache examples to disk (default False = no caching)

    Returns:
        Generator yielding lists of text documents
    """
    if not HAS_DATASETS:
        raise RuntimeError("Please install 'datasets' library: pip install datasets")

    # 配置流式会话
    configure_streaming_session()

    # 使用 streaming=True 进行在线加载
    print(f"Loading dataset in streaming mode (split={split}, cache={cache_examples})...")

    try:
        # 在线数据集只有 "train" split，对于 val split，我们使用 train 并跳过部分数据
        actual_split = "train"
        print(f"Loading from '{actual_split}' split (online dataset only has 'train')")

        # 使用 HuggingFace datasets 库进行流式加载
        dataset = load_dataset(
            DATASET_NAME,
            split=actual_split,
            streaming=True,
            download_mode="force_redownload" if not cache_examples else "reuse_cache_if_exists",
        )

        # 对数据进行迭代
        batch_size = 1024  # 每个批次的文档数
        batch = []

        for idx, example in enumerate(dataset):
            if idx % step != start % step:
                continue

            batch.append(example['text'])

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # 产出剩余的批次
        if batch:
            yield batch

    except Exception as e:
        print(f"Error during streaming: {e}")
        raise

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
