import shutil
import pandas as pd
from pyspark.sql import functions as F
from common import load_table, write_formats, winner, group_type, run_experiment


def data_type_experiment(spark, scale, data, out, work, check_content):
    rows, schemas = [], []
    for table in ['store_sales', 'customer', 'item', 'date_dim', 'store']:
        print(f'SF{scale} types: {table}', flush=True)
        frame = load_table(spark, data, table)
        count = frame.count()
        last = frame.columns[-1]
        terminal_is_null = frame.where(F.col(last).isNotNull()).limit(1).count() == 0
        if terminal_is_null:
            frame = frame.drop(last)
        groups = {}
        for field in frame.schema.fields:
            group = group_type(field.dataType)
            groups.setdefault(group, []).append(field.name)
            schemas.append(dict(table=table, column=field.name, spark_type=field.dataType.simpleString(), group=group, rows=count))
        for group, cols in groups.items():
            folder = work / f'types_sf{scale}' / table / group
            actual, values = write_formats(frame.select(*cols), folder, spark, check_content)
            assert actual == count
            pq, pq_full, pq_parts = values['parquet']
            oc, oc_full, oc_parts = values['orc']
            rows.append(dict(scale_factor=scale, table=table, data_type=group, columns=len(cols), row_count=count,
                             parquet_data_bytes=pq, orc_data_bytes=oc, parquet_directory_bytes=pq_full, orc_directory_bytes=oc_full,
                             parquet_parts=pq_parts, orc_parts=oc_parts, orc_over_parquet_data_ratio=oc/pq,
                             smaller_format_data_only=winner(pq,oc), trailing_null_removed=terminal_is_null))
            shutil.rmtree(folder)
        pd.DataFrame(rows).to_csv(out / 'type_by_table_results.csv',index=False)
        pd.DataFrame(schemas).to_csv(out / 'inferred_schema_by_column.csv',index=False)
    summary = pd.DataFrame(rows).groupby('data_type',as_index=False)[['parquet_data_bytes','orc_data_bytes','columns']].sum()
    summary['smaller_format'] = [winner(pq,oc) for pq,oc in zip(summary.parquet_data_bytes,summary.orc_data_bytes)]
    summary.to_csv(out / 'data_type_aggregate_results.csv',index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    run_experiment('types', data_type_experiment)
