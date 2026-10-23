"""Canonical same-prefix comparisons; shared root uncertainty cancels."""
import math
import statistics


def resolution(a,b,config):
    mad=max(statistics.median(abs(x-statistics.median(a['samples'])) for x in a['samples']),
            statistics.median(abs(x-statistics.median(b['samples'])) for x in b['samples']))
    return max(config['near_tie_absolute_ms'],config['near_tie_relative']*max(a['canonical_ms'],b['canonical_ms']),
               config['uncertainty_mad_multiplier']*mad)


def canonicalize(rows,base,config):
    for row in rows:row['canonical_ms']=base*math.exp(row['target'])
    pairs=[];counts={'beneficial':0,'neutral':0,'harmful':0,'identical_pairs':0,'near_tie_pairs':0}
    advance=next(r for r in rows if r['candidate_index']==0)
    for i,a in enumerate(rows):
        delta=a['canonical_ms']-advance['canonical_ms']
        neutral=a['object']==advance['object'] or abs(delta)<=resolution(a,advance,config)
        counts['neutral' if neutral else 'harmful' if delta>0 else 'beneficial']+=1
        for j,b in enumerate(rows[:i]):
            if a['object']==b['object']:counts['identical_pairs']+=1;continue
            ti,tj=a['canonical_ms'],b['canonical_ms'];threshold=resolution(a,b,config)
            if abs(ti-tj)<=threshold:counts['near_tie_pairs']+=1;continue
            slow,fast=max(ti,tj),min(ti,tj)
            gamma=math.log(slow/(fast+threshold))
            direction=math.copysign(1,ti-tj)
            assert gamma>0 and direction*(a['target']-b['target'])>=gamma
            pairs.append((i,j,direction,gamma))
    return pairs,counts
