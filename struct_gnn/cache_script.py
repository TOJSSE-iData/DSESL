from components.helpers import root_abs_path
from struct_gnn.meta_path import MetaPathDataset
import argparse
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s;%(levelname)s;%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = argparse.ArgumentParser(description='script for meta-path feature cache')

parser.add_argument("--device", type=str, default="cpu", help="Running one device")
parser.add_argument("--cache_min_id", type=int, default=None, help="cache_min_id")
parser.add_argument("--cache_max_id", type=int, default=None, help="cache_max_id")
parser.add_argument("--cache_by_gene", action="store_true", default=False, help="cache by gene, instead of gene-pair")

params = parser.parse_args()

MetaPathDataset(
    data_root=root_abs_path("data_new"), device=params.device,
).cache_all(
    min_id=params.cache_min_id,
    max_id=params.cache_max_id,
    cache_by_gene=params.cache_by_gene,
)
