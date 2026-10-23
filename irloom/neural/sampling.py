"""Seeded coverage of structural action roles, without using timing labels."""
import random
from .hierarchy import structure


def select(graph,limit,seed):
    parents,_,callees,_=structure(graph)
    strata={}
    descriptors={}
    for index,candidate in enumerate(graph['candidates']):
        if candidate['kind']=='advance':continue
        target=candidate['target'];current=target;function=-1;depth=0
        while current in parents:
            current=parents[current];node=graph['nodes'][current]
            if node['name'] in ('affine.for','scf.for'):depth+=1
            if node['kind']=='function':function=current;break
        key=(candidate['kind'],function,depth,callees.get(target,-1))
        strata.setdefault(key,[]).append(index);descriptors[index]=key
    rng=random.Random(seed)
    groups=list(strata.values());rng.shuffle(groups)
    for group in groups:rng.shuffle(group)
    chosen=[0]
    while len(chosen)<min(limit,len(graph['candidates'])):
        progress=False
        for group in groups:
            if group and len(chosen)<limit:chosen.append(group.pop());progress=True
        if not progress:break
    chosen=sorted(chosen)
    return chosen,{'seed':seed,'available':len(graph['candidates']),'selected':len(chosen),
                   'available_strata':len(strata),'selected_strata':len({descriptors[i] for i in chosen if i}),
                   'roles':{str(i):descriptors.get(i,('advance',-1,0,-1)) for i in chosen}}
