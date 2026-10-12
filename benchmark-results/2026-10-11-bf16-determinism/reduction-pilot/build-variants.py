from pathlib import Path
import subprocess,shutil,hashlib,json
repo=Path('/Users/briansam-bodden/Code/java-ai/projects/vectors')
root=Path('/tmp/bf16-reduction-variants')
source=(repo/'vectors-core/src/main/java/com/integrallis/vectors/core/BFloat16Matrix.java').read_text()
a=source.index('    int lanes = vectorSum.length();',source.index('private static float finishRow('))
b=source.index('    for (int column = vectorBound;',a)
variants={
'hybrid':'''    int lanes = vectorSum.length();
    if (lanes == 16) vectorSum = vectorSum.add(vectorSum.slice(8, vectorSum));
    if (lanes >= 8) vectorSum = vectorSum.add(vectorSum.slice(4, vectorSum));
    if (lanes >= 4) vectorSum = vectorSum.add(vectorSum.slice(2, vectorSum));
    float sum = lanes >= 2 ? vectorSum.lane(0) + vectorSum.lane(1) : vectorSum.lane(0);
''',
'narrow':'''    int lanes = vectorSum.length();
    if (lanes == 16) vectorSum = vectorSum.add(vectorSum.slice(8, vectorSum));
    if (lanes >= 8) vectorSum = vectorSum.add(vectorSum.slice(4, vectorSum));
    float sum;
    if (lanes >= 4) {
      FloatVector low = (FloatVector) vectorSum.castShape(FloatVector.SPECIES_128, 0);
      low = low.add(low.slice(2, low));
      sum = low.lane(0) + low.lane(1);
    } else {
      sum = lanes >= 2 ? vectorSum.lane(0) + vectorSum.lane(1) : vectorSum.lane(0);
    }
''',
'scalar_tree':'''    int lanes = vectorSum.length();
    if (lanes == 16) vectorSum = vectorSum.add(vectorSum.slice(8, vectorSum));
    float sum;
    if (lanes >= 8) {
      sum = ((vectorSum.lane(0) + vectorSum.lane(4)) + (vectorSum.lane(2) + vectorSum.lane(6)))
          + ((vectorSum.lane(1) + vectorSum.lane(5)) + (vectorSum.lane(3) + vectorSum.lane(7)));
    } else if (lanes >= 4) {
      sum = (vectorSum.lane(0) + vectorSum.lane(2)) + (vectorSum.lane(1) + vectorSum.lane(3));
    } else {
      sum = lanes >= 2 ? vectorSum.lane(0) + vectorSum.lane(1) : vectorSum.lane(0);
    }
'''
}
records=[]
base=Path('/tmp/vectors-bf16-slice-v1-payload/fixed.jar')
for name,body in variants.items():
 d=root/name;d.mkdir(exist_ok=True);src=d/'BFloat16Matrix.java';src.write_text(source[:a]+body+source[b:]);classes=d/'classes';classes.mkdir(exist_ok=True)
 subprocess.run(['/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin/javac','--add-modules','jdk.incubator.vector','-cp',str(base),'-d',str(classes),str(src)],check=True)
 jar=root/(name+'.jar');shutil.copy2(base,jar)
 subprocess.run(['/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin/jar','--update','--file',str(jar),'-C',str(classes),'com/integrallis/vectors/core/BFloat16Matrix.class'],check=True)
 records.append({'name':name,'javaSha256':hashlib.sha256(src.read_bytes()).hexdigest(),'jarSha256':hashlib.sha256(jar.read_bytes()).hexdigest()})
(root/'variants.json').write_text(json.dumps(records,indent=2)+'\n')
