"""Render compact C5 learning curves/tables from saved results, without inference.

Run in a separate Matplotlib3.10.8 reporting environment; no torch/CUDA needed.
"""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/evaluation/chesscog_c5')

def main():
    result=json.loads((ROOT/'train_result_v1.json').read_text());prep=json.loads((ROOT/'prepare-chesscog_result_v1.json').read_text())
    destination=Path('docs/evaluation/chesscog_c5_figures');destination.mkdir(parents=True,exist_ok=True)
    figure_path=destination/'learning_curves_v1.svg'
    if figure_path.exists():raise ValueError('Refuse figure overwrite')
    train=result['training_loss_by_epoch'];history=result['history'];initial=result['initial_validation']
    updates=[];total=0
    for row in train:total+=row['batches'];updates.append(total)
    vu=[0]+[r['updates'] for r in history];vl=[initial['validation_loss']]+[r['validation_loss'] for r in history]
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axes[0].plot(updates,[r['training_loss'] for r in train],'o-',label='Training, interval mean')
    axes[0].plot(vu,vl,'s-',label='Validation, saved endpoint')
    axes[0].set_ylabel('Weighted cross-entropy');axes[0].legend(fontsize=8)
    for metric,label in [('overall_accuracy','Overall'),('occupied_accuracy','Occupied')]:
        vals=[initial['metrics'][metric]]
        for row in history:
            cm=row['confusion_matrix'];indices=range(13) if metric=='overall_accuracy' else range(1,13)
            vals.append(sum(cm[k][k] for k in indices)/sum(sum(cm[k]) for k in indices))
        axes[1].plot(vu,[100*x for x in vals],'o-',label=label)
    axes[1].set_ylabel('Synthetic validation accuracy (%)');axes[1].set_ylim(0,100);axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Optimizer updates (batch64)');ax.grid(alpha=.2)
        if result['updates']>prep['train_boards']:ax.axvline(prep['train_boards'],color='.6',ls=':',lw=1)
    fig.suptitle('C.5 — '+('Partial' if result['partial_training'] else 'Completed')+' Chesscog pretraining; seed42',fontsize=11)
    fig.savefig(figure_path);plt.close(fig)
    table=Path('docs/evaluation/chesscog_c5_class_performance_v1.csv')
    if table.exists():raise ValueError('Refuse CSV overwrite')
    with table.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['checkpoint','class_id','class_name','support','predicted_count','precision','recall'])
        for name,v in result['validation'].items():
            for c in v['metrics']['per_class']:
                writer.writerow([name,c['class_id'],result['class_names'][c['class_id']],c['support'],c['predicted_count'],c['precision'],c['recall']])
    with (ROOT/'learning_curve_v1.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['updates','training_loss_interval_mean','validation_loss']);writer.writerow([0,'',vl[0]])
        for u,t,h in zip(updates,train,history):writer.writerow([u,t['training_loss'],h['validation_loss']])
    for name,v in result['validation'].items():
        m=v['metrics'];print(name,'overall',m['overall_accuracy'],'occupied',m['occupied_accuracy'],'emptyFP',m['empty_false_positives'],'exact',m['exact_boards'],'/',m['complete_board_denominator'])
        classes=sorted((r for r in m['per_class'] if r['support']),key=lambda r:r['recall'])
        print('Lowest recalls',[(result['class_names'][r['class_id']],r['recall']) for r in classes[:4]])
    print('Figure',figure_path,'class table',table)

if __name__=='__main__':main()
