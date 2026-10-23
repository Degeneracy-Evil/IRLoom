"""Shared content summaries and role-specific reads; tree pooling stays sparse."""
import time
import torch
from torch import nn
from .encoder import SparseBlock


def structure(graph):
    parents={}; children={}; callees={}
    for edge in graph['edges']:
        source,target,kind=edge[:3]
        if kind in ('contains','defines','argument'):
            if target in parents and parents[target]!=source: raise ValueError('multiple structural owners')
            parents[target]=source;children.setdefault(source,[]).append(target)
        elif kind=='call':callees[source]=target
    depths={0:0}
    def depth(node, visiting):
        if node in depths:return depths[node]
        if node in visiting:raise ValueError('cyclic structural ownership')
        visiting.add(node)
        result=1+depth(parents.get(node,0),visiting)
        visiting.remove(node);depths[node]=result
        return result
    for node in range(len(graph['nodes'])):depth(node,set())
    return parents,children,callees,depths


class Hierarchy(nn.Module):
    def __init__(self,hidden,heads):
        super().__init__()
        self.local=SparseBlock(hidden,heads,'structured-v2')
        self.pool=nn.Linear(hidden,hidden)
        self.norm=nn.LayerNorm(hidden)
        self.calls=nn.Linear(hidden,hidden)
        self.module=nn.Linear(hidden,hidden)

    def forward(self,x,graph):
        start=time.perf_counter()
        parents,children,callees,depths=structure(graph)
        x=self.local(x,graph)
        # CFG and recursive call edges are deliberately not tree parents.
        # Every child contributes once to a shared bottom-up summary.
        for level in range(max(depths.values()),0,-1):
            leaves=[n for n,d in depths.items() if d==level and n in parents]
            if not leaves:continue
            owner=torch.tensor([parents[n] for n in leaves]); ids=torch.unique(owner)
            pooled=torch.zeros_like(x).index_add(0,owner,x[leaves])
            counts=torch.zeros(len(x)).index_add(0,owner,torch.ones(len(leaves)))
            x=x.index_copy(0,ids,self.norm(x[ids]+self.pool(pooled[ids]/counts[ids,None])))
        if callees:
            sites=list(callees)
            x=x.index_copy(0,torch.tensor(sites),self.norm(x[sites]+self.calls(x[[callees[i] for i in sites]])))
        functions=[n['id'] for n in graph['nodes'] if n['kind']=='function']
        if functions:
            x=x.index_copy(0,torch.tensor([0]),self.norm(x[0]+self.module(x[functions].mean(0)))[None])
        return x,{'summary_s':time.perf_counter()-start,'summarized_nodes':len(children),
                  'structural_depth':max(depths.values()),'resolved_calls':len(callees)}

    def read(self,h,graph):
        start=time.perf_counter()
        parents,children,callees,_=structure(graph)
        zero=torch.zeros_like(h[0]); cached={}
        def ancestor(node,kind):
            while node in parents:
                node=parents[node]
                if graph['nodes'][node]['kind']==kind:return node
            return None
        def body(node):
            if node not in cached:
                regions=[i for i in children.get(node,[]) if graph['nodes'][i]['kind']=='region']
                cached[node]=h[regions].mean(0) if regions else None
            return cached[node]
        rows=[];masks=[]
        for candidate in graph['candidates']:
            target=candidate['target'];caller=ancestor(target,'function');region=ancestor(target,'region')
            related=None
            if candidate['kind']=='inline':
                callee=callees.get(target)
                if callee is not None:
                    contents=body(callee)
                    related=(h[callee]+contents)/2 if contents is not None else None
            elif candidate['kind']=='unroll2':related=body(target)
            elif candidate['kind']=='fuse':
                operand=candidate['parameters'].get('operand',0)
                values=[e[0] for e in graph['edges'] if e[1]==target and e[2]=='uses' and (e[3] if len(e)>3 else 0)==operand]
                producer=parents.get(values[0]) if values else None
                related=h[producer] if producer is not None else None
            elif candidate['kind']=='advance':related=h[0]
            roles=[h[target],h[caller] if caller is not None else None,related,h[region] if region is not None else None,h[0]]
            rows.append(torch.stack([v if v is not None else zero for v in roles]))
            masks.append([float(v is not None) for v in roles])
        return (torch.stack(rows),torch.tensor(masks)),{'candidate_read_s':time.perf_counter()-start,
                 'candidate_roles':['target','caller','callee-body/loop-body/producer','region','module'],
                 'missing_roles':sum(1-v for row in masks for v in row)}
