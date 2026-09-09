"""Plot observed A10 short-control scaling and retained exact-reset failures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(before, after, output):
    if ([r['case'] for r in before['rows']] != [r['case'] for r in after['rows']]
            or any(r['case']['timed_controls'] != 5 for r in after['rows'])):
        raise ValueError('Plot requires matching study cases with five timed controls')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = {'Fill-v0': '#157c91', 'Excavate-v0': '#ad4d2b'}
    for task, color in colors.items():
        for summary, style, suffix in [(before, '--', 'before'), (after, '-', 'after')]:
            rows = [r for r in summary['rows'] if r['case']['env_id'] == task]
            rows.sort(key=lambda r: r['case']['num_envs'])
            n = [r['case']['num_envs'] for r in rows]
            label = task.split('-')[0]+' '+suffix
            axes[0].plot(n, [r['aggregate_env_controls_per_second'] for r in rows], style, color=color, marker='o', label=label)
            axes[1].plot(n, [r['observed_device_used_max_bytes']/1024**3 for r in rows], style, color=color, marker='o')
    labels = [r['label'].replace('-N', '\n') for r in after['rows']]
    x = list(range(len(labels)))
    for offset, summary, key, color, label in [(-.25, before, 'unselected_changed_rows', '#aa5363', 'Untouched: before'),
            (0, after, 'unselected_changed_rows', '#157c91', 'Untouched: after'),
            (.25, after, 'selected_changed_rows', '#547599', 'Selected: after')]:
        values = [r[key] for r in summary['rows']]
        axes[2].bar([i+offset for i in x], values, width=.25, color=color, label=label)
        for i, value in zip(x, values):
            if value == 0: axes[2].text(i+offset, 1, '0', ha='center', fontsize=8, color=color)
    axes[2].set_xticks(x, labels, fontsize=8)
    axes[2].set_ylabel('Changed environment/reset rows')
    axes[2].set_title('Exact-reset failures retained')
    axes[2].legend(loc='upper left', bbox_to_anchor=(0, -.18), fontsize=8, frameon=False)
    for ax in axes[:2]:
        ax.set_xscale('log', base=2)
        ax.set_xticks([2, 8, 32], ['2', '8', '32'])
        ax.set_xlabel('Environments')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_title('Short control throughput')
    axes[0].set_ylabel('Environment controls / second')
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_title('Observed device memory')
    axes[1].set_ylabel('GiB used at snapshot boundaries')
    fig.suptitle('GPU batch reset study · Fill and Excavate · NVIDIA A10', fontsize=16, x=.05, ha='left')
    fig.text(.05, .04, 'One run per case; five timed controls. No rendering, continuous memory peak, full-task success or reference-parity claim.', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.06, right=.98, bottom=.25, top=.82, wspace=.36)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path); parser.add_argument('after', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plot(json.loads(args.before.read_text()), json.loads(args.after.read_text()), args.output)
