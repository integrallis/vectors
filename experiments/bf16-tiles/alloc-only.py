from pathlib import Path
import subprocess,time
root=Path(__file__).resolve().parent
for arm in ['tile4parallel','tile4serial']:
 c=['/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin/java','--add-modules','jdk.incubator.vector','-XX:ActiveProcessorCount=4','-Xmx512m','-cp',f'{root}/{arm}.jar:/tmp/vectors-bf16-final-expressions-v1/probes.jar:/tmp/vectors-bf16-final-expressions-v1/benchmark.jar','com.integrallis.vectors.core.BFloat16MatrixAllocationProbe']
 with (root/(arm+'-alloc.log')).open('w') as f:r=subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 print(arm,r.returncode,flush=True)
