"""Plot C7 saved losses in the existing isolated C5 Matplotlib reporting environment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/evaluation/personal_c7')

def main():
    training={k:json.loads((ROOT/(k+'_result_v1.json')).read_text()) for k in 'BC'}
    figure=Path('docs/evaluation/personal_finetuning_c7_figures');figure.mkdir(exist_ok=True)
    if (figure/'learning_curves_v1.svg').exists():raise ValueError('Fresh figure output required')
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for branch,r in training.items():
        axes[0].plot([112*(x['epoch']+1) for x in r['training_loss_by_interval']],[x['training_loss'] for x in r['training_loss_by_interval']],marker='o',label=branch)
        axes[1].plot([x['updates'] for x in r['history']],[x['validation_loss'] for x in r['history']],marker='o',label=branch)
    for ax,title in zip(axes,['Training weighted CE','Validation weighted CE']):
        ax.set_title(title);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Loss');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figure/'learning_curves_v1.svg');plt.close(fig)

if __name__=='__main__':main()
