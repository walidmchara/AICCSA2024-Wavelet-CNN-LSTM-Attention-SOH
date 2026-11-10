"""Train CNN–WNN–WLSTM attention or reference baselines on NASA discharge cells."""
from __future__ import annotations
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from aiccsa_soh.data import build_split
from aiccsa_soh.model import BatterySOH
from aiccsa_soh.metrics import evaluate
from training_recovery import EpochCheckpoint


def run(args):
    cfg = json.loads(args.config.read_text(encoding='utf8'))
    exp=cfg['experiments'][args.group]
    params=cfg['reported_architecture']
    opts=cfg['engineering_choices_not_fixed_by_the_paper']
    seed=opts['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    ds=build_split(args.data_dir,exp['train_cells'],exp['test_cell'],cfg['features'],
                   lookback=opts['lookback_cycles'],val_fraction=opts['validation_fraction'],
                   target_mode=opts['target_mode'],nominal_ah=opts['nominal_capacity_ah'])
    dims=dict(features=len(cfg['features']),variant=args.model,
              filters=params['conv_filters'],wnn_units=params['wnn_units'],
              recurrent_units=params['wlstm_units'],attention_units=params['attention_units'],
              dropout=params['dropout'])
    if args.smoke:
        dims.update(filters=12,wnn_units=24,recurrent_units=12,attention_units=12)
    model=BatterySOH(**dims)
    epochs=args.epochs or (2 if args.smoke else opts['epochs'])
    if epochs<1: raise ValueError('Epochs must be positive')
    optim=torch.optim.Adam(model.parameters(),lr=opts['learning_rate'])
    train=DataLoader(TensorDataset(*ds.train),batch_size=opts['batch_size'],shuffle=True)
    data_files = sorted(Path(args.data_dir).rglob("*.csv"))
    recovery = EpochCheckpoint(args.checkpoint_dir or args.out / "recovery", {
        "cfg":cfg,"group":args.group,"model":args.model,"epochs":epochs,
        "smoke":args.smoke,"data":[(p.as_posix(),p.stat().st_size) for p in data_files],
    })
    stateful={"model":model,"optimizer":optim}
    history=[];best=float('inf');best_state=None;start_epoch=0
    if args.resume:
        restored=recovery.restore(stateful,device="cpu")
        start_epoch=restored["epoch"];best=restored["best"]
        best_state=restored["best_state"];history=restored["history"]
        print(f"Resumed at completed epoch {start_epoch}")
    for epoch in range(start_epoch,epochs):
        model.train();tot=0;n=0
        for x,y in train:
            optim.zero_grad()
            loss=(model(x)-y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optim.step();tot+=float(loss.item())*len(x);n+=len(x)
        model.eval()
        with torch.no_grad():v=float((model(ds.val[0])-ds.val[1]).square().mean())
        if v<best:best=v;best_state=copy.deepcopy(model.state_dict())
        history.append({'epoch':epoch+1,'training_mse':tot/max(n,1),'validation_mse':v})
        recovery.save(epoch+1,stateful,{"best":best,"best_state":best_state,"history":history})
        print(f'{args.group} | {args.model} | epoch {epoch+1}/{epochs} | validation MSE={v:.6f}')
    if best_state is None: raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():pred=model(ds.test[0]).cpu().numpy()
    truth=ds.test[1].cpu().numpy()
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(history).to_csv(args.out/'training_history.csv',index=False)
    pd.DataFrame({'cycle':ds.test_cycles,'observed_soh':truth,
                  'predicted_soh':pred}).to_csv(args.out/'test_predictions.csv',index=False)
    metrics=evaluate(truth,pred)
    report={'model':args.model,'doi':cfg['paper']['doi'],'group':args.group,
            'training_cells':exp['train_cells'],'test_cell':exp['test_cell'],
            'run_type':'reduced-width software check on real measurements' if args.smoke else 'independent training experiment',
            'metrics':metrics,'data_protocol':ds.metadata,
            'paper_replication_status':'Not validated against published results',
            'architecture':dims}
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    torch.save({'model':best_state,'architecture':dims,'data_protocol':ds.metadata},args.out/'checkpoint.pt')
    return report


def main():
    ap=argparse.ArgumentParser(description='AICCSA 2024 NASA battery SOH experiments')
    ap.add_argument('--config',type=Path,default=ROOT/'configs/paper.json')
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data/processed/nasa')
    ap.add_argument('--group',choices=['24C','43C'],default='24C')
    ap.add_argument('--model',choices=['AM-CNN-WNN-WLSTM','CNN-LSTM','CNN','LSTM'],default='AM-CNN-WNN-WLSTM')
    ap.add_argument('--epochs',type=int)
    ap.add_argument('--checkpoint-dir',type=Path,help='Persistent recovery folder')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/latest')
    args=ap.parse_args()
    res=run(args)
    print(json.dumps(res['metrics'],indent=2))

if __name__=='__main__': main()
