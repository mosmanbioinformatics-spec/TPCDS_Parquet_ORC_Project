
A reduced-scale reproduction of Section 5.1.2 and a data-type extension of Section 5.1.3 of Liu et al. (2025), *Data formats in analytical DBMSs: performance trade-offs and future directions*, The VLDB Journal, 34, article 30. https://doi.org/10.1007/s00778-025-00911-1

Use Linux or Google Colab with a CPU runtime, Python, Java 17 or later, Git, bzip2, tar, internet access and sufficient disk space.

```bash
python -m pip install -r requirements.txt
python run_all.py
```
The pipeline downloads TPC-DS at SF1, SF2 and SF5, measures Parquet and ORC storage with Snappy, runs the SF1 data-type and customer sensitivity experiments, and generates statistics and figures. Results are written to `results/` and figures to `figures/`. ```

- `run_all.py`: entry point for the complete experiment and analysis.
- `src/prepare_data.py`: dataset acquisition and commit verification.
- `src/experiment.py`: storage measurements for five tables at three scales.
- `src/data_types.py`: compressed data-file sizes by inferred data type at SF1.
- `src/sensitivity.py`: effect of removing the terminal all-null field in customer at SF1.
- `src/statistical_analysis.py`: descriptive statistics, quantitative comparison and figure generation.
- `src/visualization.py`, `src/common.py`: shared plotting and experimental functions.
- `results/`: measured byte counts, schemas, runtime metadata and statistics.
- `figures/`: six figures in report order, in PNG and SVG.
- `docs/figure_captions.md`: figure captions for the research report.
- `execution_log.txt`: selected verbatim excerpts from the Colab execution output; the unabridged source log is retained in the original results archive.

References: [TPC-DS](https://www.tpc.org/tpcds/), [Apache Spark](https://spark.apache.org/), [Parquet](https://parquet.apache.org/), [ORC](https://orc.apache.org/), [Snappy](https://google.github.io/snappy/).
