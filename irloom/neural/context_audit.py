"""Actual compiler/model fixture for callee-body visibility, not a benchmark."""
import json
import torch
from .driver import Session, save
from .programs import BASE
from .interleaved import Interleaved
from .recurrent import Recurrent


def audit(name):
    torch.set_num_threads(1)
    session=Session(name)
    session.bridge=BASE/'build/bin/irloom-polygeist-bridge'
    observations=[]
    for i,callee in enumerate(('f','g','f')):
        source='''module attributes {fixture.dict = {b = 2 : i32, a = 1 : i32}} {
  func.func private @f(%x: i32) -> i32 { %v = arith.addi %x, %x : i32
    return %v : i32 }
  func.func private @g(%x: i32) -> i32 { %v = arith.muli %x, %x : i32
    return %v : i32 }
  func.func @entry(%x: i32) -> i32 { %v = func.call @CALLEE(%x) : (i32) -> i32
    return %v : i32 }
}'''.replace('CALLEE',callee)
        if i==2:source=source.replace('%x','%input').replace('%v','%result').replace('b = 2 : i32, a = 1 : i32','a = 1 : i32, b = 2 : i32')
        session.ir=session.output/f'fixture-{i}.mlir';session.ir.write_text(source)
        observations.append(session.observe('S0'))
    a,b,renamed=observations
    assert a.graph['nodes']==b.graph['nodes']
    assert a.graph['nodes']==renamed.graph['nodes'] and a.graph['edges']==renamed.graph['edges']
    differences=[(x,y) for x,y in zip(a.graph['edges'],b.graph['edges']) if x!=y]
    assert len(differences)==1 and differences[0][0][2]=='call'
    index=next(i for i,c in enumerate(a.graph['candidates']) if c['kind']=='inline')
    target=a.graph['candidates'][index]['target']
    callee=next(e[1] for e in a.graph['edges'] if e[0]==target and e[2]=='call')
    results=[]
    for seed in (17,23,47):
        for kind,cls in (('A',Interleaved),('B',Recurrent)):
            for version in ('structured-v2','hierarchical-v3'):
                torch.manual_seed(seed)
                model=cls(hidden=32,heads=4,encoder_version=version,decision_version='stage-q-v1');model.eval()
                captures={};hooks=[]
                if version=='hierarchical-v3':
                    hooks=[model.hierarchy.local.register_forward_hook(lambda m,args,out:captures.update(local=out)),
                           model.heads.role_read.register_forward_pre_hook(lambda m,args:captures.update(roles=args[0]))]
                out,state=model(a)
                if hooks:
                    roles=captures['roles'];local=captures['local']
                    role_gradient=torch.autograd.grad(out[0][index],roles,retain_graph=True)[0][index,64:96].abs().sum().item()
                    body_gradient=torch.autograd.grad(roles[index,64:96].square().sum(),local)[0]
                    from .hierarchy import structure
                    parents,_,_,_=structure(a.graph)
                    descendants=[]
                    for node in a.graph['nodes']:
                        current=node['id']
                        while current in parents and current!=callee:current=parents[current]
                        if current==callee and node['name'].startswith('arith.'):descendants.append(node['id'])
                    body_gradient=float(body_gradient[descendants].abs().sum())
                    assert role_gradient>0 and body_gradient>0
                else:role_gradient=body_gradient=None
                for hook in hooks:hook.remove()
                other,_=model(b);same,_=model(renamed)
                assert torch.equal(out[0],same[0])
                delta=float((out[0][index]-other[0][index]).abs().detach())
                if version=='structured-v2':assert delta==0
                else:assert delta>0
                results.append({'seed':seed,'model':kind,'version':version,'inline_q_delta':delta,
                                'callee_role_gradient_l1':role_gradient,'body_to_role_gradient_l1':body_gradient})
    save(session.output/'context-audit.json',{'source':'legal MLIR parsed by pinned bridge; random initialization structural acceptance only',
        'only_changed_relation':differences,'ssa_and_unordered_dictionary_invariant':True,'results':results})
    print(json.dumps({'context_audit':str(session.output/'context-audit.json')}),flush=True)
    return results
