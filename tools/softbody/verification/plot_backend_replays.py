"""Plot derived scalar replay reports; no original particle/geometry assets."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cpu', type=Path)
    parser.add_argument('gpu', type=Path)
    parser.add_argument('reference_repeats', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    cpu, gpu, repeats = [json.loads(p.read_text()) for p in (args.cpu, args.gpu, args.reference_repeats)]
    if len({r['fixture_sha256'] for r in (cpu, gpu, repeats)}) != 1:
        raise ValueError('Plot inputs describe different fixtures')
    t = [r['time_s'] for r in cpu['frames']]
    if t != [r['time_s'] for r in gpu['frames']] or any(t != [r['time_s'] for r in run['samples']] for run in repeats['runs']):
        raise ValueError('Plot sample times differ')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
    colors = {'Reference': '#252b36', 'Port / CPU PhysX': '#2463bd', 'Port / GPU PhysX': '#ba4a15'}
    for ax, key, title in zip(axes[:2], ('lifted_particles', 'spilled_particles'), ('Lifted particles', 'Spilled particles')):
        values = np.array([[s['task'][key] for s in run['samples']] for run in repeats['runs']])
        ax.fill_between(t, values.min(0), values.max(0), color='#d2d6dc', alpha=.65, label='3 reference repeats: observed range')
        for label, report, role, style in (('Reference', cpu, 'reference', '-'), ('Port / CPU PhysX', cpu, 'candidate', '--'), ('Port / GPU PhysX', gpu, 'candidate', ':')):
            ax.plot(t, [r[role][key] for r in report['frames']], style, color=colors[label], linewidth=1.8, label=label)
        ax.set(title=title, xlabel='Simulation time (s)', xlim=(0, t[-1]))
        ax.grid(axis='y', alpha=.16)
    target = cpu['final']['reference']['target_particles']
    for bound in (target-100, target+150): axes[0].axhline(bound, color='#5c6470', linewidth=.75, linestyle='--')
    axes[1].axhline(20, color='#5c6470', linewidth=.75, linestyle='--')
    for label, report, style in (('Port / CPU PhysX', cpu, '--'), ('Port / GPU PhysX', gpu, ':')):
        axes[2].plot(t, [1000*r['errors']['com_distance_m'] for r in report['frames']], style, color=colors[label], linewidth=1.8)
    axes[2].set(title='Center-of-mass separation from reference', ylabel='mm', xlabel='Simulation time (s)', xlim=(0, t[-1]))
    axes[2].grid(axis='y', alpha=.16)
    fig.suptitle('Excavate: one frozen 231-control episode', x=.04, ha='left', fontsize=17, fontweight='bold')
    fig.text(.04, .885, 'Same initial particles and controls · CUDA MPM in both port runs · Dashed horizontal lines: original task limits', color='#4f5662')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .06))
    fig.text(.04, .025, 'Diagnostic comparison only. Three reference repeats are not a confidence interval; matching success does not establish physics parity.', fontsize=9, color='#4f5662')
    fig.subplots_adjust(left=.055, right=.985, top=.78, bottom=.28, wspace=.31)
    fig.savefig(args.output, dpi=170, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
