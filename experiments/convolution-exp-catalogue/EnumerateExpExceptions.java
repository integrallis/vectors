import java.io.PrintWriter;
import java.nio.file.Path;
import java.time.Instant;

/** Enumerates the complete certified domain; compiled-candidate validation is a separate check. */
public final class EnumerateExpExceptions {
  private static final int FIRST_MAGNITUDE = Float.floatToRawIntBits(0x1p-28f);
  private static final int LAST_MAGNITUDE = Float.floatToRawIntBits(104f);
  private static final int GUARD_DOUBLE_STEPS = 8;

  public static void main(String[] args) throws Exception {
    if (args.length != 1) throw new IllegalArgumentException("Usage: EnumerateExpExceptions output.jsonl");
    long checked = 0, hits = 0, started = System.nanoTime();
    try (var out = new PrintWriter(Path.of(args[0]).toFile())) {
      for (int magnitude = FIRST_MAGNITUDE; magnitude <= LAST_MAGNITUDE; magnitude++) {
        for (int sign = 0; sign < 2; sign++) {
          int inputBits = magnitude | (sign == 0 ? 0 : Integer.MIN_VALUE);
          float input = Float.intBitsToFloat(inputBits);
          double reference = StrictMath.exp((double) input);
          long raw = Double.doubleToRawLongBits(reference);
          // In this interval every exponential is a positive, normal, finite double.
          float lower = (float) Double.longBitsToDouble(raw - GUARD_DOUBLE_STEPS);
          float upper = (float) Double.longBitsToDouble(raw + GUARD_DOUBLE_STEPS);
          if (Float.floatToRawIntBits(lower) != Float.floatToRawIntBits(upper)) {
            out.printf(
                "{\"inputBits\":\"%08x\",\"inputHex\":\"%a\","
                    + "\"referenceDoubleBits\":\"%016x\",\"strictFloatBits\":\"%08x\","
                    + "\"lowerFloatBits\":\"%08x\",\"upperFloatBits\":\"%08x\"}%n",
                inputBits, input, raw, Float.floatToRawIntBits((float) reference),
                Float.floatToRawIntBits(lower), Float.floatToRawIntBits(upper));
            out.flush();
            hits++;
          }
          checked++;
        }
        if ((magnitude & 0xffffff) == 0)
          System.out.printf("progress=%d exceptions=%d utc=%s%n", checked, hits, Instant.now());
      }
    }
    System.out.printf(
        "{\"checked\":%d,\"exceptions\":%d,\"firstMagnitudeBits\":\"%08x\","
            + "\"lastMagnitudeBitsInclusive\":\"%08x\",\"guardDoubleSteps\":%d,"
            + "\"elapsedNanos\":%d,\"java\":\"%s\"}%n",
        checked, hits, FIRST_MAGNITUDE, LAST_MAGNITUDE, GUARD_DOUBLE_STEPS,
        System.nanoTime() - started, System.getProperty("java.version"));
  }
}
