"""One frozen mixed-profile baseline; never search weights using evaluation."""
import json
import math
from pathlib import Path
import re
import statistics
import time

from .compiler import ROOT
from .real_ab import ZstdAB
from .real_sqlite_ab import SQLiteAB


def profile_records(text):
    records={}
    pattern=r'^  (.+):\n    Hash: (\S+)\n    Counters: (\d+)\n    Function count: (\d+)\n    Block counts: \[([^\]]*)\]'
    for name,cfg,n,entry,blocks in re.findall(pattern,text,re.M):
        if name.startswith(("sqlite3.c:","sqlite3")):
            values=[int(entry),*[int(v) for v in blocks.split(",") if v.strip()]]
            if len(values)!=int(n) or name in records:
                raise ValueError("ambiguous core profile structure: "+name)
            records[name]=(cfg,values)
    if "sqlite3_step" not in records or "sqlite3.c:sqlite3VdbeExec" not in records:
        raise ValueError("missing SQLite core profile records")
    return records


class SQLiteMix(SQLiteAB):
    groups=("P0","P1","Pmix")
    cases={"point":(72007,2000000),"range":(82009,200000)}
    families=("point","range","orm","cte")

    def __init__(self,args):
        super().__init__(args)
        self.prior=self.namespace
        self.namespace=self.prior.parent/"pgo-mix-v1"
        self.harnesses={}

    def mix_profile(self):
        counts={}
        inputs=[]
        historical={}
        for group in ("P0","P1"):
            training=json.loads((self.prior/group/"generate/training.json").read_text())
            profile=Path(training["profile"])
            historical[group]=training
            inputs.append(profile)
            text=self.command([self.config["profdata"],"show","--all-functions","--counts",profile],"mix-counts-"+group)
            counts[group]=profile_records(text)
            self.command([self.config["profdata"],"show","--detailed-summary",profile],"mix-summary-"+group)
        if counts['P0'].keys()!=counts['P1'].keys() or any(counts['P0'][k][0]!=counts['P1'][k][0] or
                len(counts['P0'][k][1])!=len(counts['P1'][k][1]) for k in counts['P0']):
            raise ValueError("incompatible core PGO function structure")
        directory=self.namespace/"profiles/equal-raw-counts-v1"
        directory.mkdir(parents=True,exist_ok=True)
        identity={"weights":[1,1],"inputs":[{"path":str(p),"mtime_ns":p.stat().st_mtime_ns,"bytes":p.stat().st_size} for p in inputs]}
        marker=directory/"configuration.json"
        profile=directory/"training.profdata"
        cached=marker.exists() and json.loads(marker.read_text())==identity and profile.exists()
        start=time.perf_counter()
        if not cached:
            if marker.exists():
                raise ValueError("frozen profile inputs changed; use a new namespace")
            self.command([self.config["profdata"],"merge","--num-threads=2","--failure-mode=any",
                          *[f"--weighted-input=1,{p}" for p in inputs],"-o",profile],"mix-merge")
            marker.write_text(json.dumps(identity,indent=2)+"\n")
        merged=profile_records(self.command([self.config["profdata"],"show","--all-functions","--counts",profile],"mix-counts-Pmix"))
        if merged.keys()!=counts['P0'].keys() or any(merged[k][0]!=counts['P0'][k][0] or
                merged[k][1]!=[a+b for a,b in zip(counts['P0'][k][1],counts['P1'][k][1])] for k in merged):
            raise ValueError("mixed core counters differ from exact input sums")
        counts['Pmix']=merged
        self.metadata['mix']={**identity,"profile":str(profile),"cache_hit":cached,"merge_and_verify_seconds":time.perf_counter()-start,
            "core_functions":len(merged),"core_compatible":True,"all_core_counters_exact_sum":True,
            "counts":{g:{"function_entry_total":sum(v[1][0] for v in r.values()),
                         "instrumented_counter_total":sum(sum(v[1]) for v in r.values())} for g,r in counts.items()},
            "historical_training":historical,"historical_training_cpu_seconds":sum(t['training_cpu_seconds'] for t in historical.values()),
            "new_training_cpu_seconds":0,"reason":"fixed 1:1 raw counts from two approximately equal 60 CPU-second historical windows; no normalization or evaluation tuning"}
        return profile,historical

    def build_harnesses(self):
        directory=self.namespace/"harness"
        directory.mkdir(parents=True,exist_ok=True)
        source=self.source('sqlite')/'test/speedtest1.c'
        wrapper=ROOT/'benchmarks/sqlite_results.cpp'
        marker=directory/'configuration.json'
        identity={"flags":self.native_flags,"source_mtime_ns":source.stat().st_mtime_ns,
            "wrapper_mtime_ns":wrapper.stat().st_mtime_ns,"core_mtimes":{g:(v['use']/'core.o').stat().st_mtime_ns for g,v in self.programs.items()}}
        cached=marker.exists() and json.loads(marker.read_text())==identity and (directory/'reference-audit').exists() and all((directory/(g+'-audit')).exists() and (directory/(g+'-speedtest1')).exists() for g in self.groups)
        first=len(self.events)
        start=time.perf_counter()
        links=[f"--ld-path={self.lld}","-Wl,--threads=2","-Wl,--thinlto-jobs=2"]
        if not cached:
            for kind,extra in (("native",[]),("audit",["-Dsqlite3_step=irloom_step"])):
                self.command([self.config['compiler'],*self.native_flags,"-I",self.generator,"-DSPEEDTEST_OMIT_HASH","-DSQLITE_OS_KV_OPTIONAL",*extra,
                              "-c",source,"-o",directory/(kind+'.o')],"mix-harness-"+kind)
            self.command([self.config['cxx'],*self.native_flags,"-std=c++17","-I",self.generator,"-c",wrapper,"-o",directory/'results.o'],"mix-results-wrapper")
            reference=self.core_recipe(["-O0","-DNDEBUG"])
            reference[reference.index('-o')+1]=str(directory/'reference-core.o')
            self.command([*reference,"-c"],"mix-reference-core",cwd=self.generator,timeout=600)
            for group,program in [(g,v['use']/'core.o') for g,v in self.programs.items()]+[("reference",directory/'reference-core.o')]:
                for kind in (("native","audit") if group!="reference" else ("audit",)):
                    binary=directory/(group+('-speedtest1' if kind=='native' else '-audit'))
                    objects=[directory/(kind+'.o'),program]
                    if kind=='audit':
                        objects.append(directory/'results.o')
                    self.command([self.config['cxx'],*self.native_flags,*objects,"-lm","-lz",*links,"-o",binary],"mix-link-"+group+'-'+kind,timeout=600)
            marker.write_text(json.dumps(identity,indent=2)+'\n')
        self.harnesses={g:directory/(g+'-speedtest1') for g in self.groups}
        self.audit_directory=directory
        self.metadata['harness_build']={**identity,"cache_hit":cached,"seconds":time.perf_counter()-start,"event_range":[first,len(self.events)],
            "reference_core":"same official core feature recipe, -O0 -DNDEBUG, no PGO/LTO; validation only",
            "hash":"SPEEDTEST_OMIT_HASH; no --verify; full raw results compared outside timing"}
        self.metadata['harness_sizes']={g:self.command(['/usr/lib/llvm-18/bin/llvm-size',p],"mix-harness-size-"+g) for g,p in self.harnesses.items()}

    def build(self):
        self.generate_source()
        profile,historical=self.mix_profile()
        for group,t in historical.items():
            path=self.prior/group/'use'/Path(t['profile']).parent.name
            config=json.loads((path/'configuration.json').read_text())
            if config['driver_mtime_ns']!=self.driver.stat().st_mtime_ns or config['source_mtime_ns']!=(self.generator/'sqlite3.c').stat().st_mtime_ns or config['driver_flags']!=self.native_flags or f"-fprofile-instr-use={t['profile']}" not in config['flags']:
                raise ValueError('historical core/driver changed; refusing silent baseline rebuild')
            self.programs[group]={'use':path}
            self.metadata.setdefault('reused_baselines',{})[group]={"configuration":config,"directory":str(path),
                "binary_bytes":(path/'sqlite-workload').stat().st_size,
                "llvm_size":self.command(['/usr/lib/llvm-18/bin/llvm-size',path/'sqlite-workload'],'mix-reused-size-'+group)}
        self.programs['Pmix']={'use':self.compile_program('Pmix','use',profile)}
        self.build_harnesses()
        self.metadata['protocol']={"file":"docs/REAL_PMIX_PROTOCOL.md","groups":self.groups,"families":self.families,"family_weights":[0.25]*4,
            "cases":self.cases,"official_size":60,"batches":self.args.batches,"diagnostic_only":True,"near_observed_frontier_percent":3,
            "primary_metric":"process wall seconds","oracle":"per-round per-family min of three measured programs; hindsight lower bound, not deployable",
            "stop":"no more weight/attribute search, irrespective of outcome"}
        self.metadata['programs']={g:{k:str(p) for k,p in v.items()} for g,v in self.programs.items()}

    def validate_official(self):
        records=[]
        for case in ('orm','cte'):
            reference=None
            for group in ('reference',*self.groups):
                path=self.work/(group+'-'+case+'.results')
                self.command([self.audit_directory/(group+'-audit'),'--memdb','--size','60','--testset',case],
                    'mix-validate-'+group+'-'+case,cpu=True,env={'IRLOOM_SQL_RESULTS':str(path)},timeout=300)
                if not path.exists() or path.stat().st_size==0:
                    raise ValueError('missing official query result stream')
                if reference is None:
                    reference=path
                else:
                    with reference.open('rb') as a,path.open('rb') as b:
                        while True:
                            left,right=a.read(1024*1024),b.read(1024*1024)
                            if left!=right:
                                raise ValueError('official query result differs: '+group+'/'+case)
                            if not left:
                                break
                records.append({'case':case,'group':group,'bytes':path.stat().st_size,'path':str(path),'matches_O0_reference':True})
        self.metadata['correctness']=records

    def run(self):
        self.validate_official()
        self.inspect('Pmix')
        for batch in range(self.args.batches):
            for case in self.families:
                records={}
                for group in self.groups:
                    record={'name':'sqlite-'+case,'variant':group,'batch':batch+1,'status':'collecting','samples':[],
                            'median_wall_seconds':0,'mad_wall_seconds':0}
                    records[group]=record
                    self.results.append(record)
                initial=list(self.groups) if batch%2==0 else list(reversed(self.groups))
                for round in range(self.args.repetitions+1):
                    order=initial[round%3:]+initial[:round%3]
                    for group in order:
                        label=f'mix-{case}-{group}-b{batch+1}-r{round}'
                        sample=self.sample_sql(group,case,label) if case in self.cases else self.sample(label,
                            [self.harnesses[group],'--memdb','--size','60','--testset',case])
                        if round:
                            sample['round']=round
                            records[group]['samples'].append(sample)
                    ZstdAB.aggregate(records.values(),round==self.args.repetitions)
                    self.save('measuring')
                    print(f'batch {batch+1}: {case} round {round}/{self.args.repetitions}',flush=True)

    def save(self,status,error=None):
        if status=='pass' and self.results:
            portfolio=[]
            for batch in range(1,self.args.batches+1):
                times={(r['name'],r['variant']):{s['round']:s['wall_seconds'] for s in r['samples']} for r in self.results if r['batch']==batch}
                for group in self.groups:
                    rows=[]
                    for round in range(1,self.args.repetitions+1):
                        ratios=[times['sqlite-'+c,group][round]/times['sqlite-'+c,'P0'][round] for c in self.families]
                        oracle=[min(times['sqlite-'+c,g][round] for g in self.groups) for c in self.families]
                        rows.append({'round':round,'latency_ratio_to_P0':math.prod(ratios)**0.25,
                            'worst_family_regression_percent':100*(max(ratios)-1),
                            'oracle_gap_percent':100*((math.prod(times['sqlite-'+c,group][round]/o for c,o in zip(self.families,oracle)))**0.25-1)})
                    metrics={}
                    for key in ('latency_ratio_to_P0','worst_family_regression_percent','oracle_gap_percent'):
                        med=statistics.median(r[key] for r in rows)
                        metrics[key]={'median':med,'mad':statistics.median(abs(r[key]-med) for r in rows)}
                    portfolio.append({'group':group,'batch':batch,'samples':rows,'metrics':metrics})
            self.metadata['portfolio']=portfolio
        super().save(status,error)
