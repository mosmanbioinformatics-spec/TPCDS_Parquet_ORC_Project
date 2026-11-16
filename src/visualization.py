import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def winner(pq, oc):
    return 'Parquet' if pq < oc else 'ORC' if oc < pq else 'Tie'


def save(fig, name, folder):
    fig.tight_layout()
    for ext in ['png','svg']:
        fig.savefig(folder / f'{name}.{ext}', dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def bars(ax, labels, pq, oc, ylabel, xlabel=None, log=False):
    x = np.arange(len(labels))
    ax.bar(x-.19,pq,.38,label='Parquet',color='#304a63',edgecolor='#222222',linewidth=.6)
    ax.bar(x+.19,oc,.38,label='ORC',color='#e2e4e7',edgecolor='#333333',linewidth=.7,hatch='///')
    ax.set_xticks(x,labels)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if log:
        ax.set_yscale('log')
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#d9d9d9',linewidth=.5)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='best',frameon=False)


def make_figures(b, aggregate, a, v, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':11,'legend.fontsize':9,'figure.facecolor':'white','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(6.6,3.8)); bars(ax,[f'SF{x}' for x in aggregate.scale_factor],aggregate.parquet_ratio,aggregate.orc_ratio,'Compression ratio','Scale factor'); ax.set_ylim(0, max(.52, 1.1*max(aggregate.parquet_ratio.max(), aggregate.orc_ratio.max()))); save(fig,'fig01_aggregate_compression',folder)
    names=['store_sales','customer','item','date_dim','store']
    scales = sorted(b.scale_factor.unique())
    fig,axes=plt.subplots(len(scales),1,figsize=(7.2,2.75*len(scales)),sharex=True,sharey=True,squeeze=False)
    axes=axes[:,0]
    for ax,scale in zip(axes,scales):
        sub=b[b.scale_factor==scale].set_index('table').loc[names]
        bars(ax,names,sub.parquet_ratio,sub.orc_ratio,'Compression ratio',log=True)
        ax.set_ylim(min(.035, .8*min(b.parquet_ratio.min(), b.orc_ratio.min())), max(5,1.2*max(b.parquet_ratio.max(),b.orc_ratio.max()))); ax.text(.015,.95,f'SF{scale}',transform=ax.transAxes,va='top')
        if scale!=scales[0]: ax.get_legend().remove()
    axes[-1].set_xlabel('TPC-DS table'); save(fig,'fig03_table_compression',folder)
    types=[x for x in ['Integer','FloatingPoint','String','Date'] if x in set(a.data_type)] + sorted(set(a.data_type)-{'Integer','FloatingPoint','String','Date'}); aa=a.set_index('data_type').loc[types]
    fig,ax=plt.subplots(figsize=(7,4.1)); bars(ax,['Double' if x == 'FloatingPoint' else x for x in types],aa.parquet_mib,aa.orc_mib,'Compressed data size (MiB)',log=True); save(fig,'fig05_data_type_compression',folder)
    fig,ax=plt.subplots(figsize=(6.6,3.8)); bars(ax,['Original schema','Without terminal null'],v.parquet_data_bytes/2**20,v.orc_data_bytes/2**20,'Compressed data size (MiB)'); ax.set_ylim(0, max(6.5, 1.1*max(v.parquet_data_bytes.max(),v.orc_data_bytes.max())/2**20)); save(fig,'fig06_customer_sensitivity',folder)
    fig,ax=plt.subplots(figsize=(6.6,3.8)); ax.bar([f'SF{x}' for x in aggregate.scale_factor],aggregate.parquet_savings_vs_orc_pct,color='#304a63',edgecolor='#222222',width=.48); ax.set_ylabel('Space saving relative to ORC (%)'); ax.set_xlabel('Scale factor'); ax.set_ylim(min(0, 1.1*aggregate.parquet_savings_vs_orc_pct.min()),max(20,1.1*aggregate.parquet_savings_vs_orc_pct.max())); ax.grid(axis='y',color='#d9d9d9',linewidth=.5); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False); save(fig,'fig02_space_saving',folder)

