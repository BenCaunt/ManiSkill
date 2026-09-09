"""Plot independently audited scalar study results, without particle assets."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text())
    if not result['all_declared_jobs_collected'] or any(r['execution'] != 'complete' for r in result['rows']):
        raise ValueError('Plot requires all completed numeric traces; retain failures in the study report')
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.8))
    for ax, episode in zip(axes, (1,2,3)):
        rows = {r['backend']:r for r in result['rows'] if r['episode_id']==episode}
        cpu, gpu = rows['physx_cpu'], rows['physx_cuda']
        if cpu['final_reference'] != gpu['final_reference']:
            raise ValueError('Reference outcomes differ within an episode')
        values = [cpu['final_reference'], cpu['final_candidate'], gpu['final_candidate']]
        counts = [v['lifted_particles'] for v in values]
        target = values[0]['target_particles']
        ax.bar(range(3), counts, color=['#596170','#2463bd','#ba4a15'], width=.55, zorder=3)
        for x, (count, value) in enumerate(zip(counts, values)):
            ax.text(x, count+12, str(count), ha='center', va='bottom', fontweight='bold')
        for limit in (target-100, target+150):
            ax.axhline(limit, color='#7d8796', linestyle='--', linewidth=.8)
        ax.set_ylim(0, max(max(counts), target+150)*1.19)
        labels = [name+'\n'+('success' if v['success'] else 'FAIL')+'\n'+str(v['spilled_particles'])+' spills'
                  for name,v in zip(('Reference','CPU PhysX','GPU PhysX'),values)]
        ax.set_xticks(range(3), labels)
        ax.set_title(f"Seed {cpu['seed']} · target {target}\n{cpu['particles']:,} particles · {cpu['steps']} controls", fontsize=12)
        ax.grid(axis='y', alpha=.15, zorder=0)
        ax.text(.5,-.58,'Max COM separation: '+f"{1000*cpu['max_errors']['com_distance_m']:.2f} / {1000*gpu['max_errors']['com_distance_m']:.2f} mm\n"+'(CPU / GPU versus reference)',transform=ax.transAxes,ha='center',va='top',fontsize=9,color='#4f5662')
    axes[0].set_ylabel('Final lifted particles')
    fig.suptitle('Excavate: three additional recorded episodes', x=.05, ha='left', fontsize=17, fontweight='bold')
    fig.text(.05,.89,'Identical initial states and controls per episode · CUDA MPM · Dashed lines: original amount limits',color='#4f5662')
    fig.text(.05,.025,'One observed run per engine and episode. Successful task execution does not establish calibrated physics parity.',fontsize=9,color='#4f5662')
    fig.subplots_adjust(left=.065,right=.985,top=.73,bottom=.35,wspace=.31)
    fig.savefig(args.output,dpi=170,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
