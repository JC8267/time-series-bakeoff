"""Render the saved forecast and actuals without rerunning models."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--challengers', action='store_true')
    args = parser.parse_args()
    original = Path('outputs_sls_units_labor_day_20260831_timesfm3')
    folder = Path('outputs_sls_units_challengers_20260831') if args.challengers else original
    filename = 'retrospective_challenger_forecast.csv' if args.challengers else 'daily_forecast.csv'
    data = pd.read_csv(folder / filename, parse_dates=['ds'])
    for target, label, divisor, unit in [('units', 'Units', 1000, 'thousands'),
                                          ('sales', 'Sales', 1000000, 'millions')]:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        fig.subplots_adjust(left=.085, right=.975, bottom=.19, top=.77)
        fig.text(.085, .935, f'{label}: forecast and actuals', fontsize=20, weight='bold', color='#20262d')
        fig.text(.085, .887, 'September 1–28, 2026  |  Training cutoff: August 31  |  Labor Day: September 7',
                 fontsize=10, color='#515a64')
        ax.fill_between(data.ds, data[f'{target}_p10']/divisor, data[f'{target}_p90']/divisor,
                        color='#276799', alpha=.17, linewidth=0, label='80% prediction interval')
        ax.plot(data.ds, data[f'{target}_p50']/divisor, color='#276799', linewidth=2,
                linestyle='--', label='Median forecast')
        ax.plot(data.ds, data[f'{target}_actual']/divisor, color='#20262d', linewidth=2,
                marker='o', markersize=3.5, label='Actual')
        if args.challengers:
            old = pd.read_csv(original/'daily_forecast.csv',parse_dates=['ds'])
            ax.plot(old.ds,old[f'{target}_p50']/divisor,color='#ac774a',linewidth=1.5,
                    linestyle=':',label='Original median')
        ax.axvline(pd.Timestamp('2026-09-07'), color='#8b6b3c', linewidth=1, alpha=.75)
        ax.text(pd.Timestamp('2026-09-07'), 1.02, 'Labor Day', transform=ax.get_xaxis_transform(),
                ha='center', fontsize=10, color='#6d532d')
        ax.set_ylabel(f'{label} ({unit})', fontsize=11, labelpad=10)
        ax.set_xlim(data.ds.min()-pd.Timedelta(hours=10), data.ds.max()+pd.Timedelta(hours=10))
        ax.set_xticks(pd.to_datetime(['2026-09-01','2026-09-07','2026-09-14','2026-09-21','2026-09-28']))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for side in ['left','bottom']:
            ax.spines[side].set_color('#c8cdd2')
        ax.tick_params(axis='both', labelsize=10, length=0, pad=8, colors='#515a64')
        handles, labels = ax.get_legend_handles_labels()
        order = [2,1,3,0] if args.challengers else [2,1,0]
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc='lower left', bbox_to_anchor=(.078,.06), ncol=4 if args.challengers else 3, frameon=False, fontsize=10)
        last_actual = data.loc[data[f'{target}_actual'].notna(), 'ds'].max()
        note = 'Retrospective best-fit challenger; not independently validated.' if args.challengers else '75% holiday-aware naive / 25% TimesFM 3.0.'
        fig.text(.085,.035, f'Actuals through {last_actual:%B %d}.  {note}  Non-commercial evaluation.',
                 fontsize=9, color='#515a64')
        fig.savefig(folder/f'{target}_forecast.png', dpi=180, facecolor='white')
        fig.savefig(folder/f'{target}_forecast.svg', facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    main()
