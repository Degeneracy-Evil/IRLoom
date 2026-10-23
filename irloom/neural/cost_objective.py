"""Versioned log-cost objectives; calibrated targets are feasible fixed points."""
import torch


def cost_parts(predicted, targets, margins, pairs, version='calibrated-margin-v1'):
    regression=(predicted.sub(targets).abs().sub(margins).relu().square()).mean()
    ranking=[]
    for pair in pairs:
        i,j,direction=pair[:3]
        if version=='softplus-v1':
            ranking.append(torch.nn.functional.softplus(-direction*(predicted[i]-predicted[j])))
        elif version=='calibrated-margin-v1':
            delta=targets[i]-targets[j]
            gap=(delta.abs()-margins[i]-margins[j]).clamp_min(0)
            ranking.append((gap-delta.sign()*(predicted[i]-predicted[j])).relu().square())
        elif version=='shared-root-margin-v1':
            if len(pair)!=4 or pair[3]<=0:raise ValueError('shared-root margin requires a positive canonical pair gap')
            ranking.append((pair[3]-direction*(predicted[i]-predicted[j])).relu().square())
        elif version!='interval-only-v1':raise ValueError(version)
    if version not in ('softplus-v1','calibrated-margin-v1','shared-root-margin-v1','interval-only-v1'):raise ValueError(version)
    return regression,torch.stack(ranking).mean() if ranking else predicted.new_zeros(())


def cost_loss(predicted,targets,margins,pairs,version='calibrated-margin-v1'):
    regression,ranking=cost_parts(predicted,targets,margins,pairs,version)
    return regression+ranking


def shared_root_fixture(config):
    """Actual Torch contracts, shared by objective audit and frozen-data audit."""
    import math
    from .pair_labels import canonicalize
    rows=[{'candidate_index':0,'target':0.0,'samples':[100.0]*6,'object':b'default'},
          {'candidate_index':1,'target':math.log(.97),'samples':[97.0]*6,'object':b'candidate'}]
    pairs,_=canonicalize(rows,100.0,config)
    y=torch.tensor([0.0,math.log(.97)],dtype=torch.float64)
    margins=torch.tensor([math.log(102/98)]*2,dtype=torch.float64)
    tied=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    old=cost_loss(tied,y,margins,[(1,0,-1)],'calibrated-margin-v1')
    assert old==0
    new=cost_loss(tied,y,margins,pairs,'shared-root-margin-v1')
    gradient=torch.autograd.grad(new,tied)[0]
    assert new>0 and gradient[1]>0 and gradient[0]<0
    true=y.clone().requires_grad_();truth=cost_loss(true,y,margins,pairs,'shared-root-margin-v1')
    assert truth==0 and torch.autograd.grad(truth,true)[0].abs().sum()==0
    neutral=torch.zeros(2,requires_grad=True)
    for identical in (False,True):
        fixture=[dict(rows[0]),{**rows[1],'target':0.0 if identical else math.log(.99),
            'samples':[97.0 if identical else 99.0]*6,'object':rows[0]['object'] if identical else b'near'}]
        neutral_pairs,_=canonicalize(fixture,100.0,config)
        assert not neutral_pairs
        neutral_loss=cost_loss(neutral,torch.tensor([r['target'] for r in fixture]),torch.ones(2)*.05,neutral_pairs,'shared-root-margin-v1')
        assert neutral_loss==0 and torch.autograd.grad(neutral_loss,neutral)[0].abs().sum()==0
    return {'identical_and_near_tie_zero_loss_and_gradient':True,'scalar_fixture_not_performance':{'times_ms':[100,97],'threshold_ms':2,
        'old_tied_loss':float(old.detach()),'new_tied_loss':float(new.detach()),'new_tied_gradient':gradient.tolist(),
        'gamma':pairs[0][3],'true_q_zero_loss_and_gradient':True}}


def audit(name):
    import json,math
    from .driver import CACHE,save
    from .stage_learning import prepare_groups
    output=CACHE/'runs'/name;output.mkdir(parents=True,exist_ok=False)
    d=math.log(1.09); targets=torch.tensor([-d/2,d/2],dtype=torch.float64)
    result={}
    for version in ('softplus-v1','calibrated-margin-v1','interval-only-v1'):
        truth=targets.clone().requires_grad_()
        loss=cost_loss(truth,targets,torch.zeros(2,dtype=torch.float64),[(1,0,1)],version)
        gradient=torch.autograd.grad(loss,truth)[0]
        if version=='softplus-v1':assert gradient.abs().sum()>0
        else:assert loss==0 and gradient.abs().sum()==0
        z=torch.nn.Parameter(torch.tensor(0.0,dtype=torch.float64));optimizer=torch.optim.Adam([z],lr=0.03)
        for _ in range(1200):
            predicted=torch.stack([-z/2,z/2]);objective=cost_loss(predicted,targets,torch.zeros_like(targets),[(1,0,1)],version)
            optimizer.zero_grad();objective.backward();optimizer.step()
        expected=.73457 if version=='softplus-v1' else d
        assert abs(float(z.detach())-expected)<1e-4
        result[version]={'gradient_at_true_q_l1':float(gradient.abs().sum()),'fitted_log_gap':float(z.detach()),'implied_ratio':math.exp(float(z.detach()))}
    dataset=json.loads((CACHE/'pilots/v12-stage/dataset.json').read_text())
    config={**dataset['config']['training'],'interval_version':'log-ratio-v2'}
    groups=prepare_groups(dataset,'train',config)
    for group in groups:
        y=torch.tensor([r['target'] for r in group['rows']],requires_grad=True)
        m=torch.tensor([r['margin'] for r in group['rows']])
        loss=cost_loss(y,y.detach(),m,group['pairs'])
        assert loss==0 and torch.autograd.grad(loss,y)[0].abs().sum()==0
    equal=torch.zeros(2,requires_grad=True)
    loss=cost_loss(equal,torch.zeros(2),torch.tensor([.02,.02]),[])
    assert loss==0 and torch.autograd.grad(loss,equal)[0].abs().sum()==0
    save(output/'objective-audit.json',{'two_candidate_math_not_performance':result,'real_matched_groups_at_true_q_zero_loss':len(groups),
        'identical_and_near_tie_feasible':True,'shared_root':shared_root_fixture(config)})
    print(json.dumps(result),flush=True)
