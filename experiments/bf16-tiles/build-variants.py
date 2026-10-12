from pathlib import Path
import subprocess, shutil, hashlib, json
root=Path(__file__).resolve().parent
repo=root.parent.parent
base=Path('/tmp/vectors-core-bf16-final.jar')
java=Path('/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin')
source=(root/'source-baseline.java').read_text()
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
 start=s.index('  private static void vectorMultiplyBatchRow(')
 end=s.index('  private static void requirePositive',start)
 old=s[start:end]
 old=old.replace('private static void vectorMultiplyBatchRow(', 'private static void vectorMultiplyBatchRemainder(')
 old=old.replace('      int row) {', '      int row, int firstBatch) {')
 old=old.replace('for (int batch = 0;', 'for (int batch = firstBatch;')
 a=old.index('      output[outputOffset + batch * rows + row] =')
 old=old[:a]+"""      if (sum.length() == 16) sum = sum.add(sum.rearrange(PanamaVectorUtilSupport.REDUCE_ROTATE_8));
      if (sum.length() >= 8) sum = sum.add(sum.rearrange(PanamaVectorUtilSupport.REDUCE_ROTATE_4));
      if (sum.length() >= 4) sum = sum.add(sum.rearrange(PanamaVectorUtilSupport.REDUCE_ROTATE_2));
      if (sum.length() >= 2) sum = sum.add(sum.rearrange(PanamaVectorUtilSupport.REDUCE_ROTATE_1));
"""+old[a:]
 old=old.replace('PanamaVectorUtilSupport.reduceLanesFixedTree(sum)', 'sum.lane(0)')

 signature="""      float[] input, int inputOffset, MemorySegment weight, int batchSize,
      int rows, int columns, float[] output, int outputOffset, int row"""
 body=['  private static void vectorMultiplyBatchRow(',signature+') {',
       '    int batch = 0;',f'    for (; batch + {tile} <= batchSize; batch += {tile}) {{',
       '      vectorMultiplyBatchTile(input, inputOffset + batch * columns, weight, rows, columns, output, outputOffset + batch * rows, row);',
       '    }',
       '    vectorMultiplyBatchRemainder(input, inputOffset, weight, batchSize, rows, columns, output, outputOffset, row, batch);',
       '  }',
       '  private static void vectorMultiplyBatchTile(',
       '      float[] input, int inputOffset, MemorySegment weight, int rows, int columns, float[] output, int outputOffset, int row) {',
       '    int vectorBound = columns - columns % VALUES_PER_VECTOR;',
       '    long rowOffset = (long) row * columns * Short.BYTES;']
 for i in range(tile):body +=[f'      int activation{i} = inputOffset + {i} * columns;',f'      FloatVector sum{i} = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);']
 body += ['      for (int column = 0; column < vectorBound; column += VALUES_PER_VECTOR) {','        IntVector packed = IntVector.fromMemorySegment(PanamaVectorUtilSupport.INT_SPECIES, weight, rowOffset + (long) column * Short.BYTES, ByteOrder.LITTLE_ENDIAN);','        FloatVector evenWeights = packed.lanewise(VectorOperators.LSHL, Short.SIZE).reinterpretAsFloats();','        FloatVector oddWeights = packed.and(0xffff0000).reinterpretAsFloats();']
 for i in range(tile):body +=[f'        FloatVector first{i} = FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, input, activation{i} + column);',f'        FloatVector second{i} = FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, input, activation{i} + column + PanamaVectorUtilSupport.FLOAT_SPECIES.length());',f'        sum{i} = PanamaVectorUtilSupport.fma(evenWeights, first{i}.rearrange(EVEN_INPUTS, second{i}), sum{i});',f'        sum{i} = PanamaVectorUtilSupport.fma(oddWeights, first{i}.rearrange(ODD_INPUTS, second{i}), sum{i});']
 body +=['      }']
 for i in range(tile):
  for lanes,rot in [(16,8),(8,4),(4,2),(2,1)]:
   body +=[f'    if (sum{i}.length() >= {lanes}) sum{i} = sum{i}.add(sum{i}.rearrange(PanamaVectorUtilSupport.REDUCE_ROTATE_{rot}));']
  body +=[f'    output[outputOffset + {i} * rows + row] = finishRow(weight, sum{i}.lane(0), input, activation{i}, columns, rowOffset, vectorBound);']
 body +=['  }',old]
 s=s[:start]+'\n'.join(body)+s[end:]
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
