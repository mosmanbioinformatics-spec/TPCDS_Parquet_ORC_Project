import shutil
import pandas as pd
from common import TABLES, load_table, write_formats, winner, run_experiment


def compression_experiment(spark, scale, data, out, work, check_content):
    rows = []
    for table in TABLES:
        print(f'SF{scale}: {table}', flush=True)
        frame = load_table(spark, data, table)
        folder = work / f'compression_sf{scale}' / table
        count, values = write_formats(frame, folder, spark, check_content)
        raw = (data / f'{table}.dat').stat().st_size
        pq, pq_full, pq_parts = values['parquet']
        oc, oc_full, oc_parts = values['orc']
        rows.append(dict(scale_factor=scale, table=table, rows=count, columns=len(frame.columns), raw_bytes=raw,
                         parquet_data_bytes=pq, orc_data_bytes=oc, parquet_directory_bytes=pq_full, orc_directory_bytes=oc_full,
                         parquet_parts=pq_parts, orc_parts=oc_parts, parquet_ratio=pq_full/raw, orc_ratio=oc_full/raw,
                         winner=winner(pq_full, oc_full), readback_row_count_and_types_match=True,
                         full_content_checked=check_content))
        pd.DataFrame(rows).to_csv(out / 'compression_results.csv', index=False)
        shutil.rmtree(folder)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    run_experiment('compression', compression_experiment)
