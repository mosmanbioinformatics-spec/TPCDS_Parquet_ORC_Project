import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from visualization import save, make_figures
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results'
FORMATS = ['parquet', 'orc']
PAPER_RATIOS = {'parquet': 0.33, 'orc': 0.36}
PAPER_TYPES = {'Integer': 'ORC', 'FloatingPoint': 'Parquet', 'String': 'Parquet'}


def describe(values):
    mean = values.mean()
    sd = 0.0 if values.nunique() == 1 else values.std(ddof=0)
    return dict(n_configurations=len(values), mean=mean, median=values.median(),
                population_sd=sd, min=values.min(), max=values.max(),
                range=values.max()-values.min(), cv_pct=100*sd/mean if mean else np.nan)


def compression_statistics(scales, out):
    data = pd.concat([pd.read_csv(SOURCE / f'compression_sf{s}' / 'compression_results.csv')
                      for s in scales], ignore_index=True)
    assert not data.duplicated(['scale_factor', 'table']).any()
    assert set(data.scale_factor) == set(scales)
    assert data.groupby('scale_factor').size().eq(5).all()
    assert data.readback_row_count_and_types_match.all()
    assert (data.parquet_directory_bytes >= data.parquet_data_bytes).all()
    assert (data.orc_directory_bytes >= data.orc_data_bytes).all()
    size_cols = ['raw_bytes'] + [f'{fmt}_directory_bytes' for fmt in FORMATS]
    assert data[size_cols].gt(0).all().all()
    for fmt in FORMATS:
        ratio = data[f'{fmt}_directory_bytes']/data.raw_bytes
        np.testing.assert_allclose(ratio, data[f'{fmt}_ratio'])
        data[f'{fmt}_ratio'] = ratio
    aggregate = data.groupby('scale_factor', as_index=False)[size_cols].sum()
    for fmt in FORMATS:
        aggregate[f'{fmt}_ratio'] = aggregate[f'{fmt}_directory_bytes']/aggregate.raw_bytes
    description = []
    for scope, frame in [('aggregate', aggregate), *data.groupby('table', sort=True)]:
        for fmt in FORMATS:
            description.append(dict(scope=scope, format=fmt,
                                    **describe(frame[f'{fmt}_ratio'])))
    pd.DataFrame(description).to_csv(out / 'descriptive_statistics.csv', index=False)
    paired = data[['scale_factor', 'table', *size_cols]].copy()
    paired['parquet_ratio'] = data.parquet_ratio
    paired['orc_ratio'] = data.orc_ratio
    paired['orc_minus_parquet_pp'] = 100*(data.orc_ratio-data.parquet_ratio)
    paired['orc_relative_to_parquet_pct'] = 100*(data.orc_directory_bytes/data.parquet_directory_bytes-1)
    paired['smaller_format'] = np.select([data.parquet_directory_bytes < data.orc_directory_bytes,
                                         data.orc_directory_bytes < data.parquet_directory_bytes],
                                        ['Parquet', 'ORC'], default='Tie')
    paired['smaller_format_savings_pct'] = 100*(1-data[[f'{fmt}_directory_bytes' for fmt in FORMATS]].min(axis=1)/
                                                 data[[f'{fmt}_directory_bytes' for fmt in FORMATS]].max(axis=1))
    paired['raw_share_pct'] = 100*data.raw_bytes/data.groupby('scale_factor').raw_bytes.transform('sum')
    paired.to_csv(out / 'paired_table_comparison.csv', index=False)
    wins = pd.crosstab(paired.scale_factor, paired.smaller_format).reindex(columns=['Parquet','ORC','Tie'], fill_value=0)
    wins.to_csv(out / 'format_wins.csv')
    growth = []
    for scope, frame in [('aggregate', aggregate), *data.groupby('table', sort=True)]:
        frame = frame.sort_values('scale_factor')
        first = frame.iloc[0]
        for row in frame.itertuples(index=False):
            for fmt in FORMATS:
                size = f'{fmt}_directory_bytes'
                ratio = f'{fmt}_ratio'
                growth.append(dict(scope=scope, scale_factor=row.scale_factor, reference_sf=int(first.scale_factor),
                                   format=fmt, raw_bytes=row.raw_bytes, compressed_bytes=getattr(row,size),
                                   raw_growth_factor=row.raw_bytes/first.raw_bytes,
                                   compressed_growth_factor=getattr(row,size)/first[size],
                                   compression_ratio=getattr(row,ratio),
                                   ratio_change_pp=100*(getattr(row,ratio)-first[ratio]),
                                   storage_reduction_vs_raw_pct=100*(1-getattr(row,ratio))))
    pd.DataFrame(growth).to_csv(out / 'scalability_analysis.csv', index=False)
    comparison = []
    for row in aggregate.itertuples(index=False):
        for fmt in FORMATS:
            observed = getattr(row, f'{fmt}_ratio')
            reference = PAPER_RATIOS[fmt]
            comparison.append(dict(scale_factor=row.scale_factor, format=fmt, observed_ratio=observed,
                                   approximate_paper_ratio=reference, difference_pp=100*(observed-reference),
                                   reference='Liu et al. (2025), Fig. 10, Snappy, full TPC-DS SF10; visual estimate',
                                   directly_matched_configuration=False))
    pd.DataFrame(comparison).to_csv(out / 'original_study_comparison.csv', index=False)
    aggregate['parquet_savings_vs_orc_pct'] = 100*(1-aggregate.parquet_directory_bytes/aggregate.orc_directory_bytes)
    aggregate.to_csv(out / 'aggregate_statistics.csv', index=False)
    return aggregate


def extensions(out):
    types = pd.read_csv(SOURCE / 'types_sf1/type_by_table_results.csv')
    assert not types.duplicated(['table', 'data_type']).any()
    assert types.trailing_null_removed.all()
    schema = pd.read_csv(SOURCE / 'types_sf1/inferred_schema_by_column.csv')
    groups = schema.groupby(['table','group']).agg(columns=('column','size'), min_rows=('rows','min'), max_rows=('rows','max')).reset_index().rename(columns={'group':'data_type'})
    checked = types.merge(groups, on=['table','data_type'], suffixes=('','_schema'), validate='one_to_one')
    assert len(checked) == len(types) == len(groups)
    assert checked.columns_schema.eq(checked['columns']).all()
    assert checked.min_rows.eq(checked.row_count).all() and checked.max_rows.eq(checked.row_count).all()
    sizes = [f'{fmt}_data_bytes' for fmt in FORMATS]
    assert types[sizes].gt(0).all().all()
    aggregate = types.groupby('data_type', as_index=False)[sizes+['columns']].sum()
    type_summary = pd.read_csv(SOURCE / 'types_sf1/data_type_aggregate_results.csv')
    pd.testing.assert_frame_equal(aggregate.set_index('data_type').sort_index(),
                                  type_summary.set_index('data_type')[sizes+['columns']].sort_index())
    aggregate['observed_smaller_format'] = np.select([aggregate.parquet_data_bytes < aggregate.orc_data_bytes,
                                                     aggregate.orc_data_bytes < aggregate.parquet_data_bytes],
                                                    ['Parquet','ORC'], default='Tie')
    aggregate['savings_pct'] = 100*(1-aggregate[sizes].min(axis=1)/aggregate[sizes].max(axis=1))
    aggregate['paper_smaller_format'] = aggregate.data_type.map(PAPER_TYPES).fillna('Not separately reported')
    aggregate['directional_comparison'] = np.where(aggregate.data_type.isin(PAPER_TYPES),
        np.where(aggregate.observed_smaller_format == aggregate.paper_smaller_format, 'Consistent', 'Different'),
        'Not comparable')
    aggregate.to_csv(out / 'data_type_comparison.csv', index=False)
    sensitivity = pd.read_csv(SOURCE / 'sensitivity_sf1/sensitivity_customer_terminal_column.csv').set_index('variant')
    old, new = sensitivity.loc['original_inferred'], sensitivity.loc['drop_terminal_null']
    assert old.row_count == new.row_count and old.column_count - new.column_count == 1
    rows = []
    for fmt in FORMATS:
        for measure in ['data_bytes','dir_bytes']:
            key = f'{fmt}_{measure}'
            before, after = int(old[key]), int(new[key])
            rows.append(dict(format=fmt, measure=measure, before_bytes=before, after_bytes=after,
                             reduction_bytes=before-after, reduction_pct=100*(1-after/before)))
    pd.DataFrame(rows).to_csv(out / 'sensitivity_analysis.csv', index=False)
    return aggregate


def reference_figure(aggregate, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':10,
                         'axes.labelsize':11, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(aggregate))
    for i, (fmt, color, hatch) in enumerate([('parquet','#304a63',''),('orc','#e2e4e7','///')]):
        offset = -.19 if i == 0 else .19
        ax.bar(x+offset, 100*aggregate[f'{fmt}_ratio'], .36, label='Parquet' if fmt == 'parquet' else 'ORC',
               color=color, edgecolor='#333333', hatch=hatch, linewidth=.7)
        ax.bar(len(x)+.65+offset, 100*PAPER_RATIOS[fmt], .36, color=color,
               edgecolor='#333333', hatch=hatch, alpha=.6, linewidth=.7, linestyle='--')
    ax.axvline(len(x)-.15, color='#777777', linewidth=.8, linestyle=':')
    ax.set_xticks([*x, len(x)+.65], [*[f'SF{s}' for s in aggregate.scale_factor], 'SF10 (paper, approx.)'])
    ax.set_ylabel('Compression ratio (%)')
    ax.set_ylim(0, max(50, 110*aggregate[['parquet_ratio','orc_ratio']].max().max()))
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#d9d9d9', linewidth=.5)
    ax.spines[['top','right']].set_visible(False)
    save(fig, 'fig04_original_study', folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', nargs='+', type=int, choices=[1,2,5], default=[1,2,5])
    args = parser.parse_args()
    out = SOURCE / 'statistics'
    out.mkdir(parents=True, exist_ok=True)
    aggregate = compression_statistics(sorted(set(args.scales)), out)
    types = extensions(out)
    b = pd.concat([pd.read_csv(SOURCE / f'compression_sf{s}' / 'compression_results.csv')
                   for s in sorted(set(args.scales))], ignore_index=True)
    for fmt in FORMATS:
        b[f'{fmt}_ratio'] = b[f'{fmt}_directory_bytes']/b.raw_bytes
    a = types.assign(parquet_mib=types.parquet_data_bytes/2**20, orc_mib=types.orc_data_bytes/2**20)
    v = pd.read_csv(SOURCE / 'sensitivity_sf1/sensitivity_customer_terminal_column.csv').set_index('variant')
    make_figures(b, aggregate, a, v, ROOT / 'figures')
    reference_figure(aggregate, ROOT / 'figures')
    print(aggregate.to_string(index=False))
    print(types[['data_type','observed_smaller_format','directional_comparison']].to_string(index=False))
    print(f'Statistical results saved to {out}')


if __name__ == '__main__':
    main()
