"""Versioned log-cost objectives; calibrated targets are feasible fixed points."""
import torch


def cost_loss(predicted, targets, margins, pairs, version='calibrated-margin-v1'):
    regression=(predicted.sub(targets).abs().sub(margins).relu().square()).mean()
    ranking=[]
    for i,j,direction in pairs:
        if version=='softplus-v1':
            ranking.append(torch.nn.functional.softplus(-direction*(predicted[i]-predicted[j])))
        elif version=='calibrated-margin-v1':
            delta=targets[i]-targets[j]
            gap=(delta.abs()-margins[i]-margins[j]).clamp_min(0)
            ranking.append((gap-delta.sign()*(predicted[i]-predicted[j])).relu().square())
        elif version!='interval-only-v1':raise ValueError(version)
    if version not in ('softplus-v1','calibrated-margin-v1','interval-only-v1'):raise ValueError(version)
    return regression+(torch.stack(ranking).mean() if ranking else 0)


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
    save(output/'objective-audit.json',{'two_candidate_math_not_performance':result,'real_matched_groups_at_true_q_zero_loss':len(groups),'identical_and_near_tie_feasible':True})
    print(json.dumps(result),flush=True)
