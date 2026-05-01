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
package com.integrallis.vectors.studio.web.routing;

import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.search.DocumentView;
import com.integrallis.vectors.studio.sidecart.SidecartRecord;
import com.integrallis.vectors.studio.sidecart.SidecartRegistry;
import com.integrallis.vectors.studio.sidecart.SidecartSource;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.Optional;

/**
 * Streams binary blobs (e.g. extracted images) for a document. Lookup order:
 *
 * <ol>
 *   <li>The {@link SidecartRegistry} — if a source is bound for the collection, it wins.
 *   <li>The active {@link StudioSession#backend()} (vectors-server's blob endpoint).
 * </ol>
 *
 * Content type is taken from the sidecart record when present, then from the document's metadata
 * ({@code mime}, {@code contentType}, or {@code format}), then sniffed from the first bytes.
 */
public final class BlobRoutes implements HttpService {

  private final StudioSession session;
  private final SidecartRegistry sidecart;

  public BlobRoutes(StudioSession session) {
    this(session, SidecartRegistry.empty());
  }

  public BlobRoutes(StudioSession session, SidecartRegistry sidecart) {
    this.session = session;
    this.sidecart = sidecart == null ? SidecartRegistry.empty() : sidecart;
  }

  @Override
  public void routing(HttpRules rules) {
    rules.get("/collections/{name}/blobs/{id}", this::serve);
  }

  private void serve(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    String id = req.path().pathParameters().get("id");

    // 1. Sidecart wins if bound.
    Optional<SidecartSource> src = sidecart.get(name);
    if (src.isPresent()) {
      try {
        Optional<SidecartRecord> rec = src.get().get(id);
        if (rec.isPresent() && rec.get().blob() != null) {
          byte[] bytes = rec.get().blob();
          String mime = rec.get().mime();
          send(res, bytes, mime != null ? mime : sniff(bytes));
          return;
        }
      } catch (RuntimeException e) {
        res.status(Status.BAD_GATEWAY_502).send("sidecart fetch failed: " + e.getMessage());
        return;
      }
    }

    // 2. Fall back to the studio backend.
    Optional<byte[]> blob;
    try {
      blob = session.backend().getBlob(name, id);
    } catch (IllegalArgumentException e) {
      res.status(Status.NOT_FOUND_404).send("collection not found: " + name);
      return;
    } catch (RuntimeException e) {
      res.status(Status.BAD_GATEWAY_502).send("blob fetch failed: " + e.getMessage());
      return;
    }
    if (blob.isEmpty()) {
      res.status(Status.NOT_FOUND_404).send("no blob for document: " + id);
      return;
    }
    byte[] bytes = blob.get();
    send(res, bytes, resolveContentType(name, id, bytes));
  }

  private void send(ServerResponse res, byte[] bytes, String contentType) {
    res.status(Status.OK_200);
    res.header("Content-Type", contentType);
    res.header("Cache-Control", "private, max-age=300");
    res.send(bytes);
  }

  private String resolveContentType(String collection, String id, byte[] bytes) {
    DocumentView doc = session.backend().getDocument(collection, id);
    if (doc != null && doc.metadata() != null) {
      Object mime = doc.metadata().get("mime");
      if (mime == null) mime = doc.metadata().get("contentType");
      if (mime instanceof String s && !s.isBlank()) return s;
      Object fmt = doc.metadata().get("format");
      if (fmt instanceof String s && !s.isBlank()) {
        String lower = s.toLowerCase();
        if (lower.contains("/")) return lower;
        return "image/" + (lower.equals("jpg") ? "jpeg" : lower);
      }
    }
    return sniff(bytes);
  }

  /** Magic-byte sniffing for a small set of common formats; returns octet-stream when unknown. */
  static String sniff(byte[] b) {
    if (b == null || b.length < 4) return "application/octet-stream";
    if ((b[0] & 0xff) == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G') return "image/png";
    if ((b[0] & 0xff) == 0xff && (b[1] & 0xff) == 0xd8 && (b[2] & 0xff) == 0xff)
      return "image/jpeg";
    if (b[0] == 'G' && b[1] == 'I' && b[2] == 'F' && b[3] == '8') return "image/gif";
    if (b.length >= 12
        && b[0] == 'R'
        && b[1] == 'I'
        && b[2] == 'F'
        && b[3] == 'F'
        && b[8] == 'W'
        && b[9] == 'E'
        && b[10] == 'B'
        && b[11] == 'P') return "image/webp";
    if (b[0] == '%' && b[1] == 'P' && b[2] == 'D' && b[3] == 'F') return "application/pdf";
    if (b[0] == 'O' && b[1] == 'g' && b[2] == 'g' && b[3] == 'S') return "audio/ogg";
    if (b.length >= 4 && b[0] == 'I' && b[1] == 'D' && b[2] == '3') return "audio/mpeg";
    return "application/octet-stream";
  }
}
