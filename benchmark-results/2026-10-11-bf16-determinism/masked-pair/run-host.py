"""Six process pairs across the complete BF16 shape matrix; candidates must pass all probes."""
import json,pathlib,subprocess,os,time
out=pathlib.Path('/work/out');java='/opt/jdk/bin/java';commands=[]
def run(label,command,timeout=120):
    start=time.monotonic()
    with (out/(label+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    commands.append({'label':label,'command':command,'exitCode':result.returncode,'wallSeconds':time.monotonic()-start})
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
    assert result.returncode==0,label
common=[java,'--add-modules','jdk.incubator.vector','-XX:ActiveProcessorCount=4','-Xmx512m']
for arm in ['rotate','sve_lanes','masked']:
    for bits in [64,128,256]:
        for rows,cols in [(48,5120),(5,37),(8193,37),(512,2048)]:
            run(f'{arm}-cold-{bits}-{rows}x{cols}',common+[f'-Dvectors.maxBits={bits}','-cp',arm+'.jar:probes.jar:benchmark.jar','com.integrallis.vectors.core.BFloat16MatrixColdHotProbe',str(rows),str(cols)])
    run(arm+'-allocation',common+['-cp',arm+'.jar:probes.jar:benchmark.jar','com.integrallis.vectors.core.BFloat16MatrixAllocationProbe'])
order=[]
for pair in range(6):
    arms=['baseline','rotate','sve_lanes','masked'];arms=arms[pair%4:]+arms[:pair%4]
    if pair%2:arms.reverse()
    for arm in arms:
        label=f'pair{pair}-{arm}';order.append({'pair':pair,'arm':arm,'label':label})
        (out/'order.json').write_text(json.dumps(order,indent=2)+'\n')
        (out/'STATUS').write_text(label+'\n')
        command=[java,'--add-modules','jdk.incubator.vector','-cp',arm+'.jar:benchmark.jar','org.openjdk.jmh.Main','BFloat16MatrixBenchmark.mappedBfloat16.*','-f','1','-wi','5','-i','5','-w','1s','-r','1s','-p','shape=48x5120,4864x896,151936x896','-jvmArgsAppend','-XX:ActiveProcessorCount=4 -Xmx3g','-prof','gc','-rf','json','-rff',str(out/(label+'.json'))]
        run(label,command,360)
        records=json.loads((out/(label+'.json')).read_text());assert len(records)==6 and all(r['primaryMetric']['score']>0 for r in records)
(out/'STATUS').write_text('COMPLETE\n')
