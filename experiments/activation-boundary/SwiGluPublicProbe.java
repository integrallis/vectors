package com.integrallis.vectors.core;
import java.util.Arrays;
/** Verifies the actual public bulk path against the unchanged scalar arithmetic. */
public final class SwiGluPublicProbe {
  public static void main(String[] args) {
    long samples = args.length == 0 ? 10_000_000 : Long.parseLong(args[0]);
    boolean exhaustive = args.length > 1 && args[1].equals("exhaustive");
    if (exhaustive) samples = 2L * (0x42d00000 - 0x31800000);
    float[] input = new float[4096], up = new float[4096], output = new float[4096];
    Arrays.fill(up, 1.0f);
    long seed = 417326, failures = 0;
    for (long n = 0; n < samples; n += input.length) {
      int count = (int) Math.min(input.length, samples - n);
      for (int i = 0; i < count; i++) {
        seed ^= seed << 13; seed ^= seed >>> 7; seed ^= seed << 17;
        int bits = exhaustive ? (int)(0x31800000L + (n+i)/2) | ((n+i)%2==0 ? 0 : 0x80000000) : (int)seed;
        input[i] = Float.intBitsToFloat(bits);
      }
      VectorUtil.swiGlu(output, 0, input, 0, up, 0, count);
      for (int i = 0; i < count; i++) {
        float value=input[i];
        float expected=value/(1.0f+(float)StrictMath.exp(-value));
        if (Float.floatToIntBits(expected) != Float.floatToIntBits(output[i])) {
          if (failures++ < 10) System.out.println(value+" expected="+expected+" actual="+output[i]);
        }
      }
    }
    System.out.println("samples="+samples+" failures="+failures);
    if (failures != 0) throw new AssertionError("public SwiGLU differs from scalar contract");
  }
}
