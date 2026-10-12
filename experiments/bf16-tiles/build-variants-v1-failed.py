from pathlib import Path
import subprocess, shutil, hashlib, json
root=Path(__file__).resolve().parent
repo=root.parent.parent
base=Path('/tmp/vectors-core-bf16-final.jar')
java=Path('/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin')
source=(repo.parent/'vectors/benchmark-results/2026-10-11-bf16-determinism/inline-row/direct_reduce.java').read_text()
def make(tile, parallel):
 s=source
 if parallel:
  start=s.index('    // Four interleaved rows')
  s=s[:start]+'''    if (GgufParallelSupport.willParallelize(weight, (long) rows * columns,
        GgufParallelSupport.DEFAULT_MIN_ELEMENTS)) {
      GgufParallelSupport.forEachRow(weight, rows, columns,
          row -> vectorMultiplyRows(input, inputOffset, weight, row, row + 1, columns, output, outputOffset));
      return;
    }
'''+s[start:]
 start=s.index('    for (int batch = 0; batch < batchSize; batch++)',s.index('  private static void vectorMultiplyBatchRow('))
 body=['    int batch = 0;',f'    for (; batch + {tile} <= batchSize; batch += {tile}) {{']
 for i in range(tile):body +=[f'      int activation{i} = inputOffset + (batch + {i}) * columns;',f'      FloatVector sum{i} = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);']
 body += ['      for (int column = 0; column < vectorBound; column += VALUES_PER_VECTOR) {','        IntVector packed = IntVector.fromMemorySegment(PanamaVectorUtilSupport.INT_SPECIES, weight, rowOffset + (long) column * Short.BYTES, ByteOrder.LITTLE_ENDIAN);','        FloatVector evenWeights = packed.lanewise(VectorOperators.LSHL, Short.SIZE).reinterpretAsFloats();','        FloatVector oddWeights = packed.and(0xffff0000).reinterpretAsFloats();']
 for i in range(tile):body +=[f'        FloatVector first{i} = FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, input, activation{i} + column);',f'        FloatVector second{i} = FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, input, activation{i} + column + PanamaVectorUtilSupport.FLOAT_SPECIES.length());',f'        sum{i} = PanamaVectorUtilSupport.fma(evenWeights, first{i}.rearrange(EVEN_INPUTS, second{i}), sum{i});',f'        sum{i} = PanamaVectorUtilSupport.fma(oddWeights, first{i}.rearrange(ODD_INPUTS, second{i}), sum{i});']
 body +=['      }']
 for i in range(tile):body +=[f'      output[outputOffset + (batch + {i}) * rows + row] = finishRow(weight, PanamaVectorUtilSupport.reduceLanesFixedTree(sum{i}), input, activation{i}, columns, rowOffset, vectorBound);']
 body +=['    }','    for (; batch < batchSize; batch++) {']
 s=s[:start]+'\n'.join(body)+s[start+len('    for (int batch = 0; batch < batchSize; batch++) {'):]
 return s
records=[]
for tile,parallel in [(2,True),(4,True),(4,False)]:
 name=f'tile{tile}'+('parallel' if parallel else 'serial')
 d=root/name;d.mkdir(exist_ok=True)
 src=d/'BFloat16Matrix.java';src.write_text(make(tile,parallel));classes=d/'classes';classes.mkdir(exist_ok=True)
 subprocess.run([str(java/'javac'),'--add-modules','jdk.incubator.vector','-cp',str(base),'-d',str(classes),str(src)],check=True)
 jar=root/(name+'.jar');shutil.copy2(base,jar)
 subprocess.run([str(java/'jar'),'--update','--file',str(jar),'-C',str(classes),'com/integrallis/vectors/core/BFloat16Matrix.class'],check=True)
 records.append({'name':name,'javaSha256':hashlib.sha256(src.read_bytes()).hexdigest(),'jarSha256':hashlib.sha256(jar.read_bytes()).hexdigest()})
(root/'variants.json').write_text(json.dumps(records,indent=2)+'\n')
