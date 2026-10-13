package com.integrallis.vectors.core;
import java.util.SplittableRandom;
import jdk.incubator.vector.FloatVector;
/** Actual compiled helper proof; catalogue enumeration is separate. */
public class ScalarMathProof {
 static final int[] EXCEPTIONS = {0xb3000000,0x33800000,0x36fdffc1,0x377eff81,0x383a3ef1,0x38e69cc1,0x39c6be5b,0x39e5bb1d,0xbae0e25c,0xbbf0edf1,0x3d1a274e,0x4001b249,0x40315b33,0xc16912cd,0x4288942b};
 public static void main(String[] args) {
  boolean exhaustive=args[0].equals("exhaustive");
  int first=Float.floatToRawIntBits(0x1p-28f),last=Float.floatToRawIntBits(104f);
  long count=exhaustive?2L*(last-first+1):1000000,checked=0,mismatches=0,started=System.nanoTime();
  int lanes=ReproducibleActivationExp.FLOAT_SPECIES.length();float[] input=new float[lanes],actual=new float[lanes];
  var random=new SplittableRandom(842066);int[] boundary={0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc00000,0xffc12345,0x7f7fffff,0xff7fffff,0x31800000,0xb1800000,0x42d00000,0xc2d00000,0x42b17218,0xc2cff1b5};
  for(long base=0;base<count;base+=lanes){
   for(int lane=0;lane<lanes;lane++){
    long index=base+lane;int raw;
    if(exhaustive)raw=first+(int)(index/2)|((index&1)==0?0:Integer.MIN_VALUE);
    else if(index<EXCEPTIONS.length*5)raw=EXCEPTIONS[(int)index/5]+((int)index%5-2);
    else if(index<EXCEPTIONS.length*5+boundary.length*5){int n=(int)index-EXCEPTIONS.length*5;raw=boundary[n/5]+n%5-2;}
    else raw=random.nextInt();
    input[lane]=Float.intBitsToFloat(raw);
   }
   for(int lane=0;lane<lanes;lane++)actual[lane]=ReproducibleScalarExp.exp(input[lane]);
   for(int lane=0;lane<lanes&&base+lane<count;lane++){
    float expected=(float)StrictMath.exp((double)input[lane]);
    if(Float.floatToIntBits(actual[lane])!=Float.floatToIntBits(expected)){
     if(mismatches++<12)System.out.printf("MISMATCH x=%a actual=%a expected=%a%n",input[lane],actual[lane],expected);
    }
    checked++;
   }
   if(exhaustive&&(base&0x1ffffff)==0)System.out.println("checked="+checked+" mismatches="+mismatches);
  }
  System.out.printf("{\"mode\":\"%s\",\"checked\":%d,\"mismatches\":%d,\"requestedBits\":\"%s\",\"effectiveBits\":%d,\"fma\":%s,\"elapsedNanos\":%d}%n",args[0],checked,mismatches,System.getProperty("vectors.maxBits","default"),PanamaVectorUtilSupport.FLOAT_SPECIES.vectorBitSize(),PanamaConstants.HAS_FAST_VECTOR_FMA,System.nanoTime()-started);
  if(mismatches!=0)throw new AssertionError("contract failure");
 }
}
