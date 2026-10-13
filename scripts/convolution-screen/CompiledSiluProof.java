package com.integrallis.vectors.core;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.SplittableRandom;

/** Independent validation of an actual array-only SiLU helper, not a replacement implementation. */
public final class CompiledSiluProof {
  static MethodHandle target;
  static long checked, specialMismatches, signedZeroMismatches, overPublishedTolerance, maxUlp;
  static double maxAbsolute, maxPortableAbsolute;
  static long portableBoundaryChecks, portableViolations;
  static int maxAbsInput, maxUlpInput;
  static float maxAbsActual, maxAbsExpected;
  static long hash = 0xcbf29ce484222325L;

  static long ordered(float value) {
    int bits = Float.floatToRawIntBits(value);
    return Integer.toUnsignedLong(bits < 0 ? ~bits : bits | Integer.MIN_VALUE);
  }

  static void check(float input, float actual) {
    // Literal published scalar SiLU expression; comparison is never substituted into candidate.
    float expected = input / (1.0f + (float) Math.exp(-input));
    checked++;
    // Enumerated certified ambiguities in float-cast Math.exp, with exp argument=-input.
    int expBits = Float.floatToRawIntBits(-input);
    boolean ambiguous = switch (expBits) {
      case 0xb3000000, 0x33800000, 0x36fdffc1, 0x377eff81, 0x383a3ef1,
           0x38e69cc1, 0x39c6be5b, 0x39e5bb1d, 0xbae0e25c, 0xbbf0edf1,
           0x3d1a274e, 0x4001b249, 0x40315b33, 0xc16912cd, 0x4288942b -> true;
      default -> false;
    };
    if (ambiguous) {
      long center = Double.doubleToRawLongBits(StrictMath.exp(-input));
      for (int direction : new int[] {-8,8}) {
        float e = (float) Double.longBitsToDouble(center + direction);
        float possible = input / (1f + e);
        double error = Math.abs((double)actual - possible);
        maxPortableAbsolute = Math.max(maxPortableAbsolute, error);
        if (error > 2.0e-6f) portableViolations++;
        portableBoundaryChecks++;
      }
    }
    hash = (hash ^ Float.floatToIntBits(actual)) * 0x100000001b3L;
    if (!Float.isFinite(expected) || !Float.isFinite(actual)) {
      if (Float.floatToIntBits(expected) != Float.floatToIntBits(actual)) specialMismatches++;
      return;
    }
    double absolute = Math.abs((double) actual - expected);
    if (absolute > maxAbsolute) {
      maxAbsolute = absolute; maxAbsInput = Float.floatToRawIntBits(input);
      maxAbsActual = actual; maxAbsExpected = expected;
    }
    if (absolute > 2.0e-6f) overPublishedTolerance++;
    long ulp = Math.abs(ordered(expected) - ordered(actual));
    if (ulp > maxUlp) { maxUlp = ulp; maxUlpInput = Float.floatToRawIntBits(input); }
    if (expected == 0 && actual == 0 && Float.floatToRawIntBits(expected) != Float.floatToRawIntBits(actual)) signedZeroMismatches++;
  }

  public static void main(String[] args) throws Throwable {
    // helperClass helperMethod mode; fixed signature(float[],int,int), invoked on actual JAR.
    Class<?> helper = Class.forName(args[0]);
    Method method = helper.getDeclaredMethod(args[1], float[].class, int.class, int.class);
    method.setAccessible(true); target = MethodHandles.lookup().unreflect(method);
    int lanes = PanamaVectorUtilSupport.FLOAT_SPECIES.length();
    int[] edges = {0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
      0x7f800000,0xff800000,0x7fc00000,0xffc12345,0x7f7fffff,0xff7fffff,
      0x31800000,0xb1800000,0x42d00000,0xc2d00000,0x42b17218,0xc2ae0000,
      Float.floatToRawIntBits(17.32868f),Float.floatToRawIntBits(-17.32868f),
      0x33000000,0xb3800000,0xb6fdffc1,0xb77eff81,0xb83a3ef1,
      0xb8e69cc1,0xb9c6be5b,0xb9e5bb1d,0x3ae0e25c,0x3bf0edf1,
      0xbd1a274e,0xc001b249,0xc0315b33,0x416912cd,0xc288942b};
    // First invocation contains no fallback-triggering lane: independently exercises cold polynomial.
    float[] ordinaryInput = new float[lanes * 1024];
    SplittableRandom ordinaryRandom = new SplittableRandom(981299);
    for(int i=0;i<ordinaryInput.length;i++)ordinaryInput[i]=(float)ordinaryRandom.nextDouble(-80,80);
    ordinaryInput[0]=0f;ordinaryInput[1]=-0f;ordinaryInput[2]=80f;ordinaryInput[3]=-80f;
    float[] ordinaryCold=ordinaryInput.clone();target.invokeExact(ordinaryCold,0,ordinaryCold.length);
    for(int i=0;i<ordinaryCold.length;i++)check(ordinaryInput[i],ordinaryCold[i]);
    float[] coldInput = new float[lanes * 1024];
    SplittableRandom coldRandom = new SplittableRandom(791032);
    for (int i=0;i<coldInput.length;i++) coldInput[i]=Float.intBitsToFloat(i < edges.length * 5 ? edges[i/5]+i%5-2 : coldRandom.nextInt());
    float[] cold = coldInput.clone(); target.invokeExact(cold,0,cold.length);
    for(int i=0;i<cold.length;i++)check(coldInput[i],cold[i]);
    boolean exhaustive = args[2].equals("exhaustive");
    long count = exhaustive ? 2L*(0x42d00000-0x31800000+1) : 10_000_000L;
    float[] input = new float[lanes], actual = new float[lanes];
    SplittableRandom random = new SplittableRandom(826801);
    long started = System.nanoTime();
    for(long base=0;base<count;base+=lanes) {
      for(int lane=0;lane<lanes;lane++) {
        long index=base+lane;
        int raw=exhaustive ? (0x31800000+(int)(index/2))|((index&1)==0?0:Integer.MIN_VALUE) : random.nextInt();
        input[lane]=actual[lane]=Float.intBitsToFloat(raw);
      }
      target.invokeExact(actual,0,actual.length);
      for(int lane=0;lane<lanes&&base+lane<count;lane++)check(input[lane],actual[lane]);
    }
    float[] hot = coldInput.clone(); target.invokeExact(hot,0,hot.length);
    float[] ordinaryHot=ordinaryInput.clone();target.invokeExact(ordinaryHot,0,ordinaryHot.length);
    long hotMismatches = 0;
    for(int i=0;i<ordinaryCold.length;i++)if(Float.floatToIntBits(ordinaryCold[i])!=Float.floatToIntBits(ordinaryHot[i]))hotMismatches++;
    for(int i=0;i<cold.length;i++)if(Float.floatToIntBits(cold[i])!=Float.floatToIntBits(hot[i]))hotMismatches++;
    System.out.printf(java.util.Locale.ROOT,
      "{\"mode\":\"%s\",\"domainCount\":%d,\"checkedIncludingEdges\":%d,\"maxAbsolute\":%.17g,\"maxAbsInputBits\":\"%08x\",\"maxAbsActual\":\"%a\",\"maxAbsExpected\":\"%a\",\"maxUlp\":%d,\"maxUlpInputBits\":\"%08x\",\"overPublishedTolerance\":%d,\"specialMismatches\":%d,\"signedZeroMismatches\":%d,\"coldHotMismatches\":%d,\"resultHash\":\"%016x\",\"bits\":%d,\"elapsedNanos\":%d}%n",
      args[2],count,checked,maxAbsolute,maxAbsInput,maxAbsActual,maxAbsExpected,maxUlp,maxUlpInput,overPublishedTolerance,specialMismatches,signedZeroMismatches,hotMismatches,hash,lanes*32,System.nanoTime()-started);
    System.out.printf(java.util.Locale.ROOT,"PORTABLE_EXP_BOUNDARIES checks=%d maxAbsolute=%.17g violations=%d%n",portableBoundaryChecks,maxPortableAbsolute,portableViolations);
    if(overPublishedTolerance!=0||specialMismatches!=0||signedZeroMismatches!=0||hotMismatches!=0||portableViolations!=0)throw new AssertionError("published accuracy or deterministic special-value contract failed");
  }
}
