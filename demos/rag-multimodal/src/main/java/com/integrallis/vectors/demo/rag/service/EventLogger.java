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
package com.integrallis.vectors.demo.rag.service;

import com.integrallis.vectors.demo.rag.model.LogEntry;
import com.integrallis.vectors.demo.rag.model.LogEntry.Category;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Centralized event logger for the RAG demo application.
 *
 * <p>Collects application events and notifies registered listeners. Thread-safe for use from
 * multiple threads.
 */
public class EventLogger {

  private static final int MAX_ENTRIES = 500;

  private final List<LogEntry> entries = new CopyOnWriteArrayList<>();
  private final List<Consumer<LogEntry>> listeners = new CopyOnWriteArrayList<>();

  /**
   * Logs an INFO event.
   *
   * @param category Event category
   * @param message Log message
   */
  public void info(Category category, String message) {
    log(LogEntry.info(category, message));
  }

  /**
   * Logs a DEBUG event.
   *
   * @param category Event category
   * @param message Log message
   */
  public void debug(Category category, String message) {
    log(LogEntry.debug(category, message));
  }

  /**
   * Logs a WARN event.
   *
   * @param category Event category
   * @param message Log message
   */
  public void warn(Category category, String message) {
    log(LogEntry.warn(category, message));
  }

  /**
   * Logs an ERROR event.
   *
   * @param category Event category
   * @param message Log message
   */
  public void error(Category category, String message) {
    log(LogEntry.error(category, message));
  }

  /**
   * Logs an entry and notifies all listeners.
   *
   * @param entry Log entry to add
   */
  public void log(LogEntry entry) {
    entries.add(entry);

    // Trim old entries if we exceed max
    while (entries.size() > MAX_ENTRIES) {
      entries.remove(0);
    }

    // Notify listeners
    for (Consumer<LogEntry> listener : listeners) {
      try {
        listener.accept(entry);
      } catch (Exception e) {
        // Don't let listener errors break logging
        System.err.println("EventLogger listener error: " + e.getMessage());
      }
    }

    // Print to stdout
    System.out.println(entry.formatted());
  }

  /**
   * Adds a listener to receive new log entries.
   *
   * @param listener Callback for new entries
   */
  public void addListener(Consumer<LogEntry> listener) {
    listeners.add(listener);
  }

  /**
   * Removes a listener.
   *
   * @param listener Listener to remove
   */
  public void removeListener(Consumer<LogEntry> listener) {
    listeners.remove(listener);
  }

  /**
   * Gets all log entries.
   *
   * @return Copy of all entries
   */
  public List<LogEntry> getEntries() {
    return new ArrayList<>(entries);
  }

  /** Clears all log entries. */
  public void clear() {
    entries.clear();
  }
}
