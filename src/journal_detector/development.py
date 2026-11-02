"""Score-blind execution policy and isolated-CDF epoch development helpers."""
from dataclasses import dataclass, field
import math
import numpy as np
from .trainer import Selection


@dataclass
class DevelopmentStop:
    """Stop after neither source nor isolated target DEV improves for patience."""
    patience:int=5
    source:Selection=field(default_factory=Selection)
    cdf:Selection=field(default_factory=Selection)

    def __post_init__(self):
        self.source.patience=self.patience;self.cdf.patience=self.patience

    def observe(self,source_auc,cdf_auc,epoch):
        self.source.observe(source_auc,epoch)
        return self.cdf.observe(cdf_auc,epoch)

    @property
    def stopped(self):return self.source.stopped and self.cdf.stopped

    def state_dict(self):
        return {'patience':self.patience,'source':self.source.state_dict(),'cdf':self.cdf.state_dict()}

    @classmethod
    def from_state(cls,state):
        return cls(state['patience'],Selection(**state['source']),Selection(**state['cdf']))


def select_epoch_budget(histories,development_seeds=(1007,1042)):
    """Aggregate paired epoch observations; no formal/TEST predictions accepted.

    Candidate epoch is a suggestion, not an assertion that changing cosine's
    total budget reproduces that epoch. The shortened-budget candidate must be
    verified from the identical public initialization before formal freeze.
    """
    if not histories or set(histories)-set(development_seeds):
        raise ValueError('unregistered development seed histories')
    by_epoch={}
    for seed,rows in histories.items():
        seen=set()
        for row in rows:
            epoch=row['epoch'];target=row.get('cdf_dev',{}).get('pooled_auc');source=row.get('dev',{}).get('pooled_auc')
            if epoch in seen or not isinstance(epoch,int) or epoch<1:
                raise ValueError('invalid epoch history')
            seen.add(epoch)
            if target is None or source is None or not all(math.isfinite(v) for v in (target,source)):
                raise ValueError('finite source and isolated CDF development AUC required')
            by_epoch.setdefault(epoch,{})[seed]=(target,source)
    rows=[]
    for epoch,values in sorted(by_epoch.items()):
        if set(values)==set(histories):
            rows.append({'epoch':epoch,'mean_cdf_auc':float(np.mean([x[0] for x in values.values()])),
                         'mean_source_auc':float(np.mean([x[1] for x in values.values()]))})
    if not rows:raise ValueError('no common development epoch observations')
    # Earliest exact maximum, fixed before any observed model scores.
    best=max(rows,key=lambda r:(r['mean_cdf_auc'],-r['epoch']))
    return {'candidate_best_epoch':best['epoch'],'development_seeds':sorted(histories),
            'criterion':'mean_registered_development_seed_cdf_auc_earliest_maximum',
            'cdf_development_used':True,'test_used':False,'epochs':rows,
            'candidate_requires_fresh_cosine_budget_confirmation':True}
