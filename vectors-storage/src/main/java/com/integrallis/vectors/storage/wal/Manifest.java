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
package com.integrallis.vectors.storage.wal;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Per-namespace WAL manifest serialised as JSON to {@code {namespace}/manifest.json}. Records which
 * closed segments are indexed (■) vs unindexed (◈) and the highest-assigned durable sequence
 * number.
 *
 * <p>A bespoke parser is used rather than pulling in a JSON dependency: the schema is small, fixed,
 * and produced only by this class.
 */
final class Manifest {

  final long lastSeq;
  final List<BackendWriteAheadLog.SegmentMeta> segments;

  Manifest(long lastSeq, List<BackendWriteAheadLog.SegmentMeta> segments) {
    this.lastSeq = lastSeq;
    this.segments = segments;
  }

  String toJson() {
    StringBuilder sb = new StringBuilder(64 + segments.size() * 64);
    sb.append("{\"lastSeq\":").append(lastSeq).append(",\"segments\":[");
    for (int i = 0; i < segments.size(); i++) {
      BackendWriteAheadLog.SegmentMeta s = segments.get(i);
      if (i > 0) sb.append(',');
      sb.append("{\"index\":").append(s.index);
      sb.append(",\"firstSeq\":").append(s.firstSeq);
      sb.append(",\"lastSeq\":").append(s.lastSeq);
      sb.append(",\"indexed\":").append(s.indexed);
      sb.append('}');
    }
    sb.append("]}");
    return sb.toString();
  }

  static Manifest parse(String json) throws IOException {
    try {
      Cursor c = new Cursor(json);
      c.expect('{');
      long lastSeq = 0L;
      List<BackendWriteAheadLog.SegmentMeta> segs = new ArrayList<>();
      while (true) {
        c.skipWs();
        c.expect('"');
        String key = c.readString();
        c.skipWs();
        c.expect(':');
        c.skipWs();
        switch (key) {
          case "lastSeq" -> lastSeq = c.readLong();
          case "segments" -> segs.addAll(readSegments(c));
          default -> c.skipValue();
        }
        c.skipWs();
        if (c.peek() == ',') {
          c.next();
          continue;
        }
        c.expect('}');
        break;
      }
      return new Manifest(lastSeq, segs);
    } catch (RuntimeException e) {
      throw new IOException("malformed WAL manifest: " + e.getMessage(), e);
    }
  }

  private static List<BackendWriteAheadLog.SegmentMeta> readSegments(Cursor c) {
    List<BackendWriteAheadLog.SegmentMeta> out = new ArrayList<>();
    c.expect('[');
    c.skipWs();
    if (c.peek() == ']') {
      c.next();
      return out;
    }
    while (true) {
      c.skipWs();
      c.expect('{');
      int index = 0;
      long firstSeq = 0;
      long lastSeq = 0;
      boolean indexed = false;
      while (true) {
        c.skipWs();
        c.expect('"');
        String key = c.readString();
        c.skipWs();
        c.expect(':');
        c.skipWs();
        switch (key) {
          case "index" -> index = (int) c.readLong();
          case "firstSeq" -> firstSeq = c.readLong();
          case "lastSeq" -> lastSeq = c.readLong();
          case "indexed" -> indexed = c.readBool();
          default -> c.skipValue();
        }
        c.skipWs();
        if (c.peek() == ',') {
          c.next();
          continue;
        }
        c.expect('}');
        break;
      }
      out.add(new BackendWriteAheadLog.SegmentMeta(index, firstSeq, lastSeq, indexed));
      c.skipWs();
      if (c.peek() == ',') {
        c.next();
        continue;
      }
      c.expect(']');
      return out;
    }
  }
}
