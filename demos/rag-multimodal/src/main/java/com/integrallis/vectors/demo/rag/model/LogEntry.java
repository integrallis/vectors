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
package com.integrallis.vectors.demo.rag.model;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

/**
 * Represents a log entry in the application event log.
 *
 * @param timestamp When the event occurred
 * @param level Log level (INFO, DEBUG, WARN, ERROR)
 * @param category Event category (SYSTEM, CACHE, RETRIEVAL, LLM)
 * @param message Human-readable message
 */
public record LogEntry(Instant timestamp, Level level, Category category, String message) {

  private static final DateTimeFormatter TIME_FORMAT =
      DateTimeFormatter.ofPattern("HH:mm:ss.SSS").withZone(ZoneId.systemDefault());

  public enum Level {
    INFO,
    DEBUG,
    WARN,
    ERROR
  }

  public enum Category {
    SYSTEM("SYS"),
    CACHE("CACHE"),
    RETRIEVAL("RAG"),
    LLM("LLM"),
    PDF("PDF");

    private final String shortName;

    Category(String shortName) {
      this.shortName = shortName;
    }

    public String getShortName() {
      return shortName;
    }
  }

  /**
   * Creates an INFO level log entry.
   *
   * @param category Event category
   * @param message Log message
   * @return New log entry
   */
  public static LogEntry info(Category category, String message) {
    return new LogEntry(Instant.now(), Level.INFO, category, message);
  }

  /**
   * Creates a DEBUG level log entry.
   *
   * @param category Event category
   * @param message Log message
   * @return New log entry
   */
  public static LogEntry debug(Category category, String message) {
    return new LogEntry(Instant.now(), Level.DEBUG, category, message);
  }

  /**
   * Creates a WARN level log entry.
   *
   * @param category Event category
   * @param message Log message
   * @return New log entry
   */
  public static LogEntry warn(Category category, String message) {
    return new LogEntry(Instant.now(), Level.WARN, category, message);
  }

  /**
   * Creates an ERROR level log entry.
   *
   * @param category Event category
   * @param message Log message
   * @return New log entry
   */
  public static LogEntry error(Category category, String message) {
    return new LogEntry(Instant.now(), Level.ERROR, category, message);
  }

  /**
   * Formats the timestamp for display.
   *
   * @return Formatted time string
   */
  public String formattedTime() {
    return TIME_FORMAT.format(timestamp);
  }

  /**
   * Formats the full log entry for display.
   *
   * @return Formatted log string
   */
  public String formatted() {
    return String.format(
        "[%s] [%s] [%s] %s", formattedTime(), level, category.getShortName(), message);
  }
}
