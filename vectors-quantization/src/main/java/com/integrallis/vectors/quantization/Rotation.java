/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.quantization;

/**
 * Pluggable rotation strategy for rotation-based quantizers (RaBitQ, TurboQuant). The rotation maps
 * vectors to a space where per-coordinate quantization produces better distance estimators by
 * decorrelating coordinate dependencies.
 *
 * <p>All implementations must satisfy:
 *
 * <ul>
 *   <li><b>Distance preservation</b>: {@code ||R*a - R*b|| == ||a - b||} for all vectors a, b
 *   <li><b>Round-trip</b>: {@code inverseRotate(rotate(v)) ≈ v} (within floating-point tolerance)
 *   <li><b>Determinism</b>: Same seed produces identical rotation parameters
 * </ul>
 *
 * <p>Implementations differ in computational cost and decorrelation quality:
 *
 * <table>
 * <caption>Rotation strategy comparison (d=128)</caption>
 * <tr><th>Strategy</th><th>FMAs/rotate</th><th>Parameters</th><th>Quality</th></tr>
 * <tr><td>{@link RandomRotation}</td><td>16,384</td><td>d²</td><td>Baseline</td></tr>
 * <tr><td>{@link QuaternionRotation}</td><td>1,024</td><td>4 per block</td><td>Best</td></tr>
 * <tr><td>{@link GivensRotation}</td><td>256</td><td>2 per pair</td><td>Competitive</td></tr>
 * </table>
 *
 * @see RandomRotation
 * @see GivensRotation
 * @see QuaternionRotation
 */
public interface Rotation {

  /**
   * Applies the rotation to a vector.
   *
   * @param vector the input vector (must have length == {@link #dimension()})
   * @return a new array containing the rotated vector
   * @throws IllegalArgumentException if vector length != dimension
   */
  float[] rotate(float[] vector);

  /**
   * Applies the inverse rotation (recovers original coordinates).
   *
   * @param vector the rotated vector (must have length == {@link #dimension()})
   * @return a new array containing the unrotated vector
   * @throws IllegalArgumentException if vector length != dimension
   */
  float[] inverseRotate(float[] vector);

  /** Returns the dimension this rotation operates on. */
  int dimension();
}
