import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType,
    StringType, DateType, DecimalType, TimestampType,
)

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/alamb/tpcds-data.git'
COMMITS = {
    1: 'b41f5010514222d94e50127762868aec11f6e4c5',
    2: 'eb40d5fe5ee2731bcb054bcdf2f63500a8b5c8bf',
    5: 'be462d01193c748652d8b21efff1910faad4edb0',
}
TABLES = ['customer', 'item', 'store', 'date_dim', 'store_sales']
FORMATS = ['parquet', 'orc']


def get_data(scale, data_root):
    folder = data_root / f'tpcds_sf{scale}'
    folder.parent.mkdir(parents=True, exist_ok=True)
    if not folder.exists():
        subprocess.run(['git', 'clone', '--single-branch', '--branch',
                        f'sf{scale}', '--depth', '1', REPO, str(folder)], check=True)
    commit = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMITS[scale]:
        raise RuntimeError(f'SF{scale}: expected {COMMITS[scale]}, found {commit}')
    data = folder / 'data'
    if not all((data / f'{name}.dat').is_file() and (data / f'{name}.dat').stat().st_size > 0 for name in TABLES):
        subprocess.run(['bash', '-c', 'set -o pipefail; cat data.tar.bz2.* | bzip2 -d | tar -xf -'], cwd=folder, check=True)
    if not all((data / f'{name}.dat').is_file() and (data / f'{name}.dat').stat().st_size > 0 for name in TABLES):
        raise RuntimeError(f'Incomplete dataset: {data}')
    return data, commit

def sizes(folder):
    files = [p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()]
    parts = [p for p in files if p.name.startswith('part-')]
    if not parts:
        raise RuntimeError(f'No data files written to {folder}')
    return sum(p.stat().st_size for p in parts), sum(p.stat().st_size for p in files), len(parts)

def winner(pq, orc):
    return 'Parquet' if pq < orc else 'ORC' if orc < pq else 'Tie'

def write_formats(frame, folder, spark, check_content=False):
    result = {}
    expected = frame.count()
    for fmt in FORMATS:
        path = folder / fmt
        frame.write.mode('overwrite').option('compression', 'snappy').format(fmt).save(str(path))
        data_bytes, directory_bytes, parts = sizes(path)
        back = spark.read.format(fmt).load(str(path))
        count = back.count()
        if count != expected or [(f.name, f.dataType) for f in back.schema] != [(f.name, f.dataType) for f in frame.schema]:
            raise RuntimeError(f'Row count or schema mismatch: {path}')
        if check_content and (frame.exceptAll(back).limit(1).count() or back.exceptAll(frame).limit(1).count()):
            raise RuntimeError(f'Content mismatch: {path}')
        result[fmt] = (data_bytes, directory_bytes, parts)
    return expected, result

def load_table(spark, data, table):
    return spark.read.csv(str(data / f'{table}.dat'), sep='|', inferSchema=True)

def group_type(t):
    for types, name in [((ByteType, ShortType, IntegerType, LongType), 'Integer'),
                        ((FloatType, DoubleType), 'FloatingPoint'), ((StringType,), 'String'),
                        ((DateType,), 'Date'), ((DecimalType,), 'Decimal'), ((TimestampType,), 'Timestamp')]:
        if isinstance(t, types):
            return name
    return 'Other'

def start_spark():
    spark = SparkSession.builder.appName('TPCDSStorageEvaluation').config('spark.sql.shuffle.partitions', '5').getOrCreate()
    if spark.version != '4.0.5':
        spark.stop()
        raise RuntimeError(f'Expected Spark 4.0.5, found {spark.version}')
    return spark


def save_metadata(spark, scale, commit, out, check_content):
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(scale_factor=scale, dataset_repository=REPO, dataset_commit=commit,
                tables=TABLES, python=platform.python_version(), pandas=pd.__version__,
                spark=spark.version, cpu_count=os.cpu_count(),
                java=spark.sparkContext._jvm.java.lang.System.getProperty('java.version'),
                spark_master=spark.sparkContext.master, os=platform.platform(),
                codec='snappy', schema='inferred', full_content_checked=check_content)
    if Path('/proc/meminfo').exists():
        meta['memory'] = Path('/proc/meminfo').read_text().splitlines()[0]
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


def experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, choices=list(COMMITS), default=1)
    parser.add_argument('--check-content', action='store_true')
    return parser.parse_args()


def run_experiment(name, experiment):
    args = experiment_args()
    data, commit = get_data(args.scale, ROOT / 'work/data')
    out = ROOT / 'results' / f'{name}_sf{args.scale}'
    spark = start_spark()
    try:
        save_metadata(spark, args.scale, commit, out, args.check_content)
        experiment(spark, args.scale, data, out, ROOT / 'work/generated', args.check_content)
    finally:
        spark.stop()
