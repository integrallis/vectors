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
package com.integrallis.vectors.optimizer.persist;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.integrallis.vectors.optimizer.study.TrialResult;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Stream;

/**
 * Append-only JSON-Lines persistence for study results. Each study writes to {@code
 * {root}/{studyId}.jsonl}; one line per {@link TrialResult}. Concurrent writers contend on a
 * per-file {@link ReentrantLock} so a single JVM can run several studies in parallel without
 * interleaving lines. Survives JVM restarts: trial files are flushed and durable on every
 * {@link #appendTrial} call.
 */
public final class StudyStore {

  /** Lightweight summary returned by {@link #listStudies()}. */
  public record StudySummary(String studyId, int trialCount, Instant lastModified) {}

  private final Path root;
  private final ObjectMapper mapper;
  private final ConcurrentHashMap<String, ReentrantLock> locks = new ConcurrentHashMap<>();

  /** Creates a store rooted at {@code ~/.vectors/optimizer/studies}. */
  public static StudyStore defaultRoot() {
    String home = System.getProperty("user.home", ".");
    return new StudyStore(Path.of(home, ".vectors", "optimizer", "studies"));
  }

  public StudyStore(Path root) {
    this.root = Objects.requireNonNull(root, "root");
    this.mapper = newMapper();
    try {
      Files.createDirectories(root);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to create study store at " + root, ioe);
    }
  }

  /** Appends a single trial result to the study's JSON-Lines file. Thread-safe per study. */
  public void appendTrial(String studyId, TrialResult result) {
    Objects.requireNonNull(studyId, "studyId");
    Objects.requireNonNull(result, "result");
    Path file = trialsPath(studyId);
    ReentrantLock lock = locks.computeIfAbsent(studyId, k -> new ReentrantLock());
    lock.lock();
    try {
      String line = mapper.writeValueAsString(result) + "\n";
      Files.writeString(
          file,
          line,
          StandardCharsets.UTF_8,
          StandardOpenOption.CREATE,
          StandardOpenOption.APPEND);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to append trial to " + file, ioe);
    } finally {
      lock.unlock();
    }
  }

  /** Reads every trial recorded for {@code studyId}, in append order. */
  public List<TrialResult> loadAll(String studyId) {
    Objects.requireNonNull(studyId, "studyId");
    Path file = trialsPath(studyId);
    if (!Files.exists(file)) return List.of();
    List<TrialResult> out = new ArrayList<>();
    try (var lines = Files.lines(file, StandardCharsets.UTF_8)) {
      lines.forEach(
          line -> {
            if (line.isBlank()) return;
            try {
              out.add(mapper.readValue(line, TrialResult.class));
            } catch (IOException ioe) {
              throw new UncheckedIOException("Corrupt trial line in " + file, ioe);
            }
          });
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to read trials from " + file, ioe);
    }
    return out;
  }

  /** Lists every study with at least one trial, ordered by most-recent first. */
  public List<StudySummary> listStudies() {
    if (!Files.isDirectory(root)) return List.of();
    List<StudySummary> out = new ArrayList<>();
    try (Stream<Path> children = Files.list(root)) {
      children
          .filter(p -> p.toString().endsWith(".jsonl"))
          .forEach(p -> out.add(summaryOf(p)));
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to list studies in " + root, ioe);
    }
    out.sort(Comparator.comparing(StudySummary::lastModified).reversed());
    return out;
  }

  /** Writes the per-study {@code meta.json} sidecar (descriptive only; not round-tripped). */
  public void writeMeta(String studyId, Map<String, Object> meta) {
    Objects.requireNonNull(studyId, "studyId");
    Objects.requireNonNull(meta, "meta");
    Path file = root.resolve(studyId + ".meta.json");
    try {
      Files.writeString(file, mapper.writeValueAsString(meta), StandardCharsets.UTF_8);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to write meta for " + studyId, ioe);
    }
  }

  private Path trialsPath(String studyId) {
    return root.resolve(studyId + ".jsonl");
  }

  private StudySummary summaryOf(Path p) {
    String name = p.getFileName().toString();
    String studyId = name.substring(0, name.length() - ".jsonl".length());
    int count = 0;
    try (var lines = Files.lines(p, StandardCharsets.UTF_8)) {
      count = (int) lines.filter(s -> !s.isBlank()).count();
    } catch (IOException ignored) {
      // best-effort; surface zero count if read fails
    }
    Instant when;
    try {
      when = Files.getLastModifiedTime(p).toInstant();
    } catch (IOException ioe) {
      when = Instant.EPOCH;
    }
    return new StudySummary(studyId, count, when);
  }

  private static ObjectMapper newMapper() {
    ObjectMapper m = new ObjectMapper();
    m.registerModule(new JavaTimeModule());
    m.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
    m.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
    return m;
  }
}
