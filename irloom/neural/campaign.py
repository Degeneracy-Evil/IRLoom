"""One cached entry for the frozen G1–G3 campaign; compiler interfaces are reused."""
import json
import torch
from .driver import ROOT,CACHE,save


def run(args):
    torch.set_num_threads(1)
    config=json.loads((args.campaign_config.resolve() if args.campaign_config else ROOT/'experiments/architecture/campaign-v1.json').read_text())
    output=CACHE/'campaigns'/args.name;output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('campaign protocol mismatch; use a new name')
    if not path.exists():save(path,config)
    if args.campaign=='diagnose':
        from .campaign_training import diagnose
        diagnose(output,config)
    else:
        from .campaign_corpus import probe,collect,audit
        catalog=probe(output,config)
        if args.campaign=='probe':return
        if args.campaign=='collect':
            audit(output,collect(output,config,catalog),config,catalog)
            return
        from .campaign_training import diagnose,train
        diagnosis=diagnose(output,config)
        dataset=collect(output,config,catalog)
        audit(output,dataset,config,catalog)
        models=train(output,dataset,config,diagnosis)
        if args.campaign in ('all','evaluate'):
            from .campaign_evaluation import evaluate
            evaluation=evaluate(output,config,dataset,models,catalog)
            from .campaign_summary import summarize
            summarize(output,config,catalog,dataset,diagnosis,models,evaluation)
    print(json.dumps({'campaign':str(output),'phase':args.campaign}),flush=True)
