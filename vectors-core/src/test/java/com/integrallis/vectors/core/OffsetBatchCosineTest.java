package com.integrallis.vectors.core;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.*;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class OffsetBatchCosineTest {
  @Test
  void offsetsPreserveRowOrderPaddingDuplicatesAndTails() {
    var random=new SplittableRandom(7353);
    try(var arena=Arena.ofConfined()) {
      for(int dim:new int[]{1,7,16,31,32,65,100,129,512,768}) {
        long stride=(long)(dim+13)*4;
        var matrix=arena.allocate(stride*9+64);
        float[] query=new float[dim];
        for(int d=0;d<dim;d++)query[d]=(float)random.nextDouble(-1,1);
        long[] offsets=new long[9];
        MemorySegment[] rows=new MemorySegment[9];
        for(int r=0;r<9;r++) {
          offsets[r]=64+(r*5%9)*stride;
          rows[r]=matrix.asSlice(offsets[r],(long)dim*4);
          for(int d=0;d<dim;d++)rows[r].setAtIndex(ValueLayout.JAVA_FLOAT,d,(float)random.nextDouble(-1,1));
        }
        offsets[7]=offsets[2];rows[7]=rows[2];
        for(int count=0;count<=9;count++) {
          float[] expected=new float[10],actual=new float[10];
          expected[9]=actual[9]=42f;
          VectorUtil.batchCosine(query,rows,dim,expected,count);
          VectorUtil.batchCosineWithQueryNorm(query,matrix,offsets,dim,VectorUtil.batchCosineQueryNorm(query),actual,count);
          assertArrayEquals(expected,actual,2e-6f);
          assertEquals(42f,actual[9]);
        }
        float[] actual=new float[1];
        assertThrows(IndexOutOfBoundsException.class,()->VectorUtil.batchCosineWithQueryNorm(query,matrix,new long[]{-4},dim,1f,actual,1));
        assertThrows(IndexOutOfBoundsException.class,()->VectorUtil.batchCosineWithQueryNorm(query,matrix,new long[]{Long.MAX_VALUE},dim,1f,actual,1));
      }
    }
  }
}
