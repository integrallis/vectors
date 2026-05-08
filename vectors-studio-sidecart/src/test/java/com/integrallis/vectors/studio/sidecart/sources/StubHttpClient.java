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
package com.integrallis.vectors.studio.sidecart.sources;

import java.io.IOException;
import java.net.Authenticator;
import java.net.CookieHandler;
import java.net.ProxySelector;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandler;
import java.net.http.HttpResponse.PushPromiseHandler;
import java.net.http.WebSocket;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executor;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLParameters;

/**
 * Test {@link HttpClient} that records every {@link HttpRequest} and returns canned response bodies
 * in FIFO order. Only {@link #send(HttpRequest, BodyHandler)} is implemented; everything else
 * throws {@link UnsupportedOperationException}.
 */
final class StubHttpClient extends HttpClient {

  final List<HttpRequest> requests = new java.util.ArrayList<>();
  final List<String> bodies = new java.util.ArrayList<>();
  private final Queue<String> responses = new ConcurrentLinkedQueue<>();
  private int statusCode = 200;

  StubHttpClient enqueue(String body) {
    responses.add(body);
    return this;
  }

  StubHttpClient withStatus(int code) {
    this.statusCode = code;
    return this;
  }

  @Override
  public <T> HttpResponse<T> send(HttpRequest request, BodyHandler<T> responseBodyHandler) {
    requests.add(request);
    bodies.add(captureBody(request));
    String body = responses.poll();
    if (body == null) body = "{\"success\":true,\"result\":[]}";
    @SuppressWarnings("unchecked")
    HttpResponse<T> resp = (HttpResponse<T>) new StubResponse(request, statusCode, body);
    return resp;
  }

  @Override
  public <T> CompletableFuture<HttpResponse<T>> sendAsync(
      HttpRequest request, BodyHandler<T> responseBodyHandler) {
    return CompletableFuture.completedFuture(send(request, responseBodyHandler));
  }

  @Override
  public <T> CompletableFuture<HttpResponse<T>> sendAsync(
      HttpRequest request,
      BodyHandler<T> responseBodyHandler,
      PushPromiseHandler<T> pushPromiseHandler) {
    return sendAsync(request, responseBodyHandler);
  }

  private static String captureBody(HttpRequest request) {
    return request
        .bodyPublisher()
        .map(
            pub -> {
              StringBodySubscriber sub = new StringBodySubscriber();
              pub.subscribe(sub);
              return sub.result();
            })
        .orElse("");
  }

  // ─── Unused HttpClient surface ────────────────────────────────────────────

  @Override
  public Optional<CookieHandler> cookieHandler() {
    return Optional.empty();
  }

  @Override
  public Optional<Duration> connectTimeout() {
    return Optional.empty();
  }

  @Override
  public Redirect followRedirects() {
    return Redirect.NEVER;
  }

  @Override
  public Optional<ProxySelector> proxy() {
    return Optional.empty();
  }

  @Override
  public SSLContext sslContext() {
    throw new UnsupportedOperationException();
  }

  @Override
  public SSLParameters sslParameters() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Optional<Authenticator> authenticator() {
    return Optional.empty();
  }

  @Override
  public Version version() {
    return Version.HTTP_1_1;
  }

  @Override
  public Optional<Executor> executor() {
    return Optional.empty();
  }

  @Override
  public WebSocket.Builder newWebSocketBuilder() {
    throw new UnsupportedOperationException();
  }

  // ─── Inner stub types ────────────────────────────────────────────────────

  private record StubResponse(HttpRequest request, int statusCode, String body)
      implements HttpResponse<String> {
    @Override
    public java.net.URI uri() {
      return request.uri();
    }

    @Override
    public java.net.http.HttpHeaders headers() {
      return java.net.http.HttpHeaders.of(java.util.Map.of(), (a, b) -> true);
    }

    @Override
    public Optional<HttpResponse<String>> previousResponse() {
      return Optional.empty();
    }

    @Override
    public Optional<javax.net.ssl.SSLSession> sslSession() {
      return Optional.empty();
    }

    @Override
    public Version version() {
      return Version.HTTP_1_1;
    }
  }

  private static final class StringBodySubscriber
      implements java.util.concurrent.Flow.Subscriber<java.nio.ByteBuffer> {
    private final StringBuilder sb = new StringBuilder();
    private final java.util.concurrent.CountDownLatch done =
        new java.util.concurrent.CountDownLatch(1);

    @Override
    public void onSubscribe(java.util.concurrent.Flow.Subscription s) {
      s.request(Long.MAX_VALUE);
    }

    @Override
    public void onNext(java.nio.ByteBuffer buf) {
      byte[] arr = new byte[buf.remaining()];
      buf.get(arr);
      sb.append(new String(arr, java.nio.charset.StandardCharsets.UTF_8));
    }

    @Override
    public void onError(Throwable t) {
      done.countDown();
    }

    @Override
    public void onComplete() {
      done.countDown();
    }

    String result() {
      try {
        done.await(2, java.util.concurrent.TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      return sb.toString();
    }
  }

  // Suppress unused-import warning on IOException — kept for future error-path tests.
  @SuppressWarnings("unused")
  private static IOException placeholder() {
    return new IOException();
  }
}
