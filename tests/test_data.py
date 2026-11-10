from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from aiccsa_soh.data import load_cell,build_split

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/nasa'
FEATURES=['voltage_mean','voltage_std','current_mean','current_std','temperature_mean','discharge_duration_s','ambient_temperature']


def test_real_csv_cell():
    x,y,cy,cap=load_cell(DATA/'B0005.csv',FEATURES)
    assert len(y)>=100
    assert len(x)==len(y)==len(cy)
    assert np.isclose(y[0],cap/2.0)


def test_data_partition_and_shapes():
    ds=build_split(DATA,['B0005','B0006','B0007'],'B0018',FEATURES)
    assert ds.train[0].shape[1:]==(16,len(FEATURES))
    assert ds.test[0].shape[0]==len(ds.test_cycles)
    assert ds.metadata['train_cells']==['B0005','B0006','B0007']
    assert ds.metadata['test_cell']=='B0018'
    assert ds.metadata['feature_window_ends_before_target_cycle']


def test_second_experiment():
    ds=build_split(DATA,['B0029','B0031','B0032'],'B0030',FEATURES)
    assert ds.test[0].shape[0]>=10


def test_battery_holdout_cannot_overlap():
    with pytest.raises(ValueError):
        build_split(DATA,['B0005','B0018'],'B0018',FEATURES)


def test_no_same_cycle_target_leakage():
    with pytest.raises(ValueError):
        load_cell(DATA/'B0005.csv',FEATURES+['capacity_ah'])


def test_training_only_scaling_and_imputation(tmp_path):
    for cid, delta in [('B0005',0),('B0006',0),('B0018',20000)]:
        arr={'cycle':list(range(1,61)),'capacity_ah':np.linspace(2.,1.5,60)}
        arr.update({f:np.linspace(1.,2.,60)+delta for f in FEATURES})
        pd.DataFrame(arr).to_csv(tmp_path/f'{cid}.csv',index=False)
    ds=build_split(tmp_path,['B0005','B0006'],'B0018',FEATURES,lookback=8)
    assert all(x<3 for x in ds.metadata['fit_mins'])
    # Holdout-only shift should remain visible; should NOT change scaling learned on training cells.
    assert float(ds.test[0].mean())>100
