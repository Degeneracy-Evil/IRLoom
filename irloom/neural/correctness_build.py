"""Relink a private bridge with one patched, exact-revision LLVM translation unit."""
import json
from pathlib import Path
import shlex
import time

from irloom.compiler import ROOT,command


def build():
    base=ROOT/'benchmark-data/architecture/frontends';revision='26eb4285b56edd8c897642078d91f16ff0fd3472'
    source=base/f'llvm-{revision}/mlir/lib/Dialect/Affine/Utils/LoopUtils.cpp'
    target=ROOT/'benchmark-data/architecture/compilers/cleanup-positional-v1'
    target.mkdir(parents=True,exist_ok=True);(target/'bin').mkdir(exist_ok=True)
    logs=ROOT/'logs/architecture/cleanup-positional-v1-build';logs.mkdir(parents=True,exist_ok=True)
    patch=ROOT/'compiler/diagnostics/affine-cleanup-positional.patch';executable=target/'bin/irloom-polygeist-bridge'
    record=target/'build.json'
    if record.exists():
        old=json.loads(record.read_text())
        if old['patch']!=patch.read_text():raise RuntimeError('patch changed; use a new compiler revision')
        if executable.exists():
            build_inspector(target,base,logs)
            return
        raise RuntimeError('private compiler cache incomplete')
    if not source.is_file():raise FileNotFoundError('matching historical Polygeist SDK/source is required')
    start=time.perf_counter();events=[];copy=target/'LoopUtils.cpp';copy.write_bytes(source.read_bytes())
    command(['patch','--batch','--forward',copy,'-i',patch],logs/'patch.log',events,'patch-private-copy')
    def flags(path):
        values={}
        for line in path.read_text().splitlines():
            if ' = ' in line:
                key,value=line.split(' = ',1);values[key]=shlex.split(value)
        return values['CXX_DEFINES']+values['CXX_INCLUDES']+values['CXX_FLAGS']
    sdk=base/'build'
    options=flags(sdk/'tools/mlir/lib/Dialect/Affine/Utils/CMakeFiles/obj.MLIRAffineUtils.dir/flags.make')
    obj=target/'LoopUtils.o'
    command(['clang++-18',*options,'-c',copy,'-o',obj],logs/'compile.log',events,'compile-patched-loop-utils',timeout=240)
    link_base=sdk/'tools/irloom'
    argv=shlex.split((link_base/'CMakeFiles/irloom-polygeist-bridge.dir/link.txt').read_text())
    for index,value in enumerate(argv):
        if index and argv[index-1]=='-o':argv[index]=str(executable)
        elif value.startswith(('CMakeFiles/','../../')):argv[index]=str((link_base/value).resolve())
    # Defining the whole pinned TU before its historical archive means the old
    # LoopUtils.cpp.o is never extracted. Other libraries/headers stay matched.
    argv.insert(1,str(obj));argv.insert(2,'-Wl,-Map='+str(target/'link.map'))
    command(argv,logs/'link.log',events,'link-private-bridge',timeout=240)
    link_map=(target/'link.map').read_text()
    assert 'libMLIRAffineUtils.a(LoopUtils.cpp.o)' not in link_map
    result={'revision':'cleanup-positional-v1','llvm_revision':revision,'source':str(source),
        'patch':patch.read_text(),'old_bridge_object_reused':True,'old_loop_utils_archive_member_extracted':False,
        'events':events,'wall_s':time.perf_counter()-start,'scope':'private static override from identical pinned source/headers; historical SDK, sources, binaries and archives unchanged'}
    record.write_text(json.dumps(result,indent=2)+'\n')
    build_inspector(target,base,logs)


def build_inspector(target,base,logs):
    exe=target/'bin/inspect-loops'
    if exe.exists():return
    flags_file=base/'build/tools/irloom/CMakeFiles/irloom-polygeist-bridge.dir/flags.make'
    options=[]
    for line in flags_file.read_text().splitlines():
        if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):options+=shlex.split(line.split(' = ',1)[1])
    obj=target/'inspect-loops.o';events=[]
    command(['clang++-18',*options,'-c',ROOT/'compiler/diagnostics/inspect-loops.cpp','-o',obj],logs/'inspect-compile.log',events,'compile-inspector',timeout=240)
    link_base=base/'build/tools/irloom'
    args=shlex.split((link_base/'CMakeFiles/irloom-polygeist-bridge.dir/link.txt').read_text())
    for i,value in enumerate(args):
        if i and args[i-1]=='-o':args[i]=str(exe)
        elif value.endswith('main.cpp.o'):args[i]=str(obj)
        elif value.startswith('../../'):args[i]=str((link_base/value).resolve())
    args.insert(1,str(target/'LoopUtils.o'))
    command(args,logs/'inspect-link.log',events,'link-inspector',timeout=240)
    (target/'inspector-build.json').write_text(json.dumps({'events':events},indent=2)+'\n')


if __name__=='__main__':build()
