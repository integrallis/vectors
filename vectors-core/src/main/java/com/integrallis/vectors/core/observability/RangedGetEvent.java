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
package com.integrallis.vectors.core.observability;

import jdk.jfr.Category;
import jdk.jfr.DataAmount;
import jdk.jfr.Description;
import jdk.jfr.Enabled;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * JDK Flight Recorder event for a single object-storage <b>ranged GET</b> — the per-query I/O unit
 * of the object-storage index. Recording these makes the query path self-describing under JFR: the
 * per-GET latency distribution, byte volume, and count-per-query are all visible in any JFR tool
 * (JMC, {@code jfr print}) with the ≈1% overhead JFR is designed for — no metrics backend required.
 *
 * <p>Pure {@code jdk.jfr}: no third-party dependency, no per-call allocation when a recording is
 * not active. Emit around a ranged GET as:
 *
 * <pre>{@code
 * RangedGetEvent event = new RangedGetEvent();
 * event.begin();                 // cheap no-op when disabled
 * byte[] bytes = backend.getRange(key, offset, length);
 * if (event.shouldCommit()) {    // enabled AND over threshold
 *   event.key = key;
 *   event.offset = offset;
 *   event.length = length;
 *   event.commit();
 * }
 * }</pre>
 *
 * <p>Disabled by default so it never adds overhead in production unless explicitly enabled (a
 * recording with {@code com.integrallis.vectors.RangedGet#enabled=true}, or a settings file). At
 * scale (many GETs/s) enable with a {@code threshold} to capture only the slow tail.
 */
@Name("com.integrallis.vectors.RangedGet")
@Label("Object-Storage Ranged GET")
@Category({"Vectors", "Object Storage"})
@Description("A ranged GET fetching one vector's bytes from object storage during two-pass rerank")
@StackTrace(false)
@Enabled(false)
public final class RangedGetEvent extends Event {

  @Label("Object Key")
  public String key;

  @Label("Offset")
  @DataAmount
  public long offset;

  @Label("Length")
  @DataAmount
  public int length;
}
