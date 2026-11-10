"""Battery-disjoint NASA PCoE discharge-cycle preprocessing."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch


@dataclass
class BatterySplit:
    train: tuple[torch.Tensor, torch.Tensor]
    val: tuple[torch.Tensor, torch.Tensor]
    test: tuple[torch.Tensor, torch.Tensor]
    test_cycles: np.ndarray
    metadata: dict


def load_cell(path: Path, features: list[str], target_mode='nominal', nominal_ah=2.0):
    df = pd.read_csv(path).sort_values('cycle', kind='stable').reset_index(drop=True)
    forbidden = {'capacity_ah', 'target', 'soh_nominal', 'soh_first_observed', 'remaining_observed_cycles'}
    if set(features) & forbidden:
        raise ValueError(f'Capacity and SOH labels cannot be input features: {set(features) & forbidden}')
    need = ['cycle', 'capacity_ah', *features]
    if set(need) - set(df):
        raise ValueError(f'{path.name}: missing columns {sorted(set(need) - set(df))}')
    if len(df) < 20 or df['cycle'].duplicated().any():
        raise ValueError(f'{path.name}: not enough independent, ordered cycles')
    cap = pd.to_numeric(df['capacity_ah'], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(cap).all() or (cap <= 0).any():
        raise ValueError(f'{path.name}: invalid discharge capacity')
    if target_mode == 'nominal':
        if nominal_ah <= 0:
            raise ValueError('nominal_ah must be positive')
        target = cap / nominal_ah
    elif target_mode == 'first_observed':
        target = cap / cap[0]
    else:
        raise ValueError('Unsupported SOH target definition')
    x = df[features].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    return x, target.astype(np.float32), df['cycle'].to_numpy(), cap[0]


def build_split(directory: Path, train_cells: list[str], test_cell: str,
                features: list[str], lookback=16, val_fraction=.20,
                target_mode='nominal', nominal_ah=2.0):
    if test_cell in train_cells or not train_cells or len(train_cells) != len(set(train_cells)):
        raise ValueError('The held-out cell must be distinct from all development batteries')
    if not 0.05 <= val_fraction < .5 or lookback < 4:
        raise ValueError('Invalid validation fraction or context length')
    group = {cid: load_cell(directory / f'{cid}.csv', features, target_mode, nominal_ah)
             for cid in [*train_cells, test_cell]}
    cut = {cid: int(len(group[cid][0]) * (1 - val_fraction)) for cid in train_cells}
    if any(c <= lookback for c in cut.values()):
        raise ValueError('Insufficient training cycles')
    fit_x = np.vstack([group[cid][0][:cut[cid]] for cid in train_cells])
    med = np.nanmedian(fit_x, axis=0)
    if not np.isfinite(med).all():
        raise ValueError('Feature missing throughout all training cells')
    fit_x = np.where(np.isfinite(fit_x), fit_x, med)
    mins = fit_x.min(axis=0)
    ranges = np.maximum(fit_x.max(axis=0) - mins, 1e-8)
    arrays = {k: ([], []) for k in ('train', 'val', 'test')}
    test_cycles = []
    for cell_id, (values, y, cycles, _) in group.items():
        x = (np.where(np.isfinite(values), values, med) - mins) / ranges
        for index in range(lookback, len(values)):
            role = 'test' if cell_id == test_cell else ('train' if index < cut[cell_id] else 'val')
            arrays[role][0].append(x[index-lookback:index].astype(np.float32))
            arrays[role][1].append(y[index])
            if role == 'test':
                test_cycles.append(int(cycles[index]))
    def as_tensors(role):
        xx, yy = arrays[role]
        if not xx:
            raise ValueError(f'Empty {role} partition')
        return torch.tensor(np.stack(xx), dtype=torch.float32), torch.tensor(yy, dtype=torch.float32)
    meta = {
        'train_cells': train_cells, 'test_cell': test_cell, 'lookback': lookback,
        'feature_names': features, 'validation_fraction': val_fraction,
        'train_cutoffs_by_cell': cut, 'fit_medians': med.tolist(),
        'fit_mins': mins.tolist(), 'fit_ranges': ranges.tolist(),
        'target_definition': ('capacity_ah / nominal_capacity_ah' if target_mode == 'nominal'
                              else 'capacity_ah / first_observed_capacity_ah'),
        'nominal_ah_assumption': nominal_ah if target_mode == 'nominal' else None,
        'feature_window_ends_before_target_cycle': True,
        'data_resolution': 'cycle-level aggregates extracted from NASA MATLAB discharge traces'
    }
    return BatterySplit(as_tensors('train'), as_tensors('val'), as_tensors('test'),
                        np.asarray(test_cycles), meta)
