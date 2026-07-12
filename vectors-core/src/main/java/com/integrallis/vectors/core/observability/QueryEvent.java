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
import jdk.jfr.Description;
import jdk.jfr.Enabled;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;

/**
 * JDK Flight Recorder event spanning one top-level ANN <b>search</b>. Its duration is the
 * end-to-end query latency; the fields capture the request shape (k, ef, over-query) and the
 * object-storage work (rerank GETs and bytes fetched). Correlate with {@link RangedGetEvent}s on
 * the same thread/time window for the per-GET breakdown.
 *
 * <p>Pure {@code jdk.jfr}, {@link Enabled disabled by default}. Emit around a search:
 *
 * <pre>{@code
 * QueryEvent event = new QueryEvent();
 * event.begin();
 * SearchOutcome out = ...;         // run the two-pass search
 * if (event.shouldCommit()) {
 *   event.k = k; event.efSearch = ef; event.overQueryFactor = over;
 *   event.results = out.ordinals().length;
 *   event.getsIssued = gets; event.bytesFetched = bytes;  // if tracked
 *   event.commit();
 * }
 * }</pre>
 */
@Name("com.integrallis.vectors.Query")
@Label("Vector Search")
@Category({"Vectors", "Query"})
@Description("A top-level ANN search: navigate on RAM codes, then object-storage rerank")
@StackTrace(false)
@Enabled(false)
public final class QueryEvent extends Event {

  @Label("k")
  public int k;

  @Label("ef Search")
  public int efSearch;

  @Label("Over-Query Factor")
  public float overQueryFactor;

  @Label("Results Returned")
  public int results;

  @Label("Ranged GETs Issued")
  public int getsIssued;

  @Label("Bytes Fetched")
  public long bytesFetched;
}
