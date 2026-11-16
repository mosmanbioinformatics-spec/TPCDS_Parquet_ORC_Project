import shutil
import pandas as pd
from pyspark.sql import functions as F
from common import load_table, write_formats, run_experiment


def sensitivity_experiment(spark, scale, data, out, work, check_content):
    customer = load_table(spark, data, 'customer')
    last = customer.columns[-1]
    if customer.where(F.col(last).isNotNull()).limit(1).count():
        raise RuntimeError('Terminal customer column is not empty')
    sensitivity = []
    for name, frame in [('original_inferred',customer),('drop_terminal_null',customer.drop(last))]:
        folder = work / f'sensitivity_sf{scale}' / name
        count, values = write_formats(frame, folder, spark, check_content)
        pq, pq_full, _ = values['parquet']
        oc, oc_full, _ = values['orc']
        sensitivity.append(dict(variant=name,row_count=count,column_count=len(frame.columns),parquet_data_bytes=pq,parquet_dir_bytes=pq_full,orc_data_bytes=oc,orc_dir_bytes=oc_full))
        shutil.rmtree(folder)
    pd.DataFrame(sensitivity).to_csv(out / 'sensitivity_customer_terminal_column.csv',index=False)
    print(pd.DataFrame(sensitivity).to_string(index=False))


if __name__ == '__main__':
    run_experiment('sensitivity', sensitivity_experiment)
