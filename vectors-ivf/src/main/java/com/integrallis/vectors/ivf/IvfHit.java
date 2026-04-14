package com.integrallis.vectors.ivf;

/**
 * A single result from an {@link IvfIndex} search.
 *
 * @param ordinal row index in the original {@code float[][] vectors} array passed to {@link
 *     IvfIndex#build}
 * @param id document identifier supplied at build time; {@code null} if no ids were provided
 * @param score similarity score; higher is more similar (dot product or negated L2)
 */
public record IvfHit(int ordinal, String id, float score) implements Comparable<IvfHit> {

  @Override
  public int compareTo(IvfHit other) {
    // Descending score: higher score = better hit = earlier in list
    return Float.compare(other.score, this.score);
  }
}
