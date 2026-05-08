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
package com.integrallis.demos.studio;

import java.util.List;

/**
 * Built-in 24-document corpus across four topics ({@code food}, {@code programming}, {@code
 * sports}, {@code music}) used by the {@link R2CorpusSeederDemo}. Lifted from {@code vectors-ivf}'s
 * test corpus so the same realistic semantic distribution drives the cloud demo.
 */
public final class Corpus {

  private Corpus() {}

  /** Returns the built-in corpus. */
  public static List<Doc> realistic() {
    return List.of(
        // food
        d("food-01", "food", "Sushi is a Japanese dish made with vinegared rice and seafood."),
        d("food-02", "food", "Italian pasta carbonara uses eggs, pancetta, and pecorino cheese."),
        d("food-03", "food", "A good ramen broth simmers for hours to develop deep umami flavor."),
        d("food-04", "food", "French croissants are laminated dough with butter folded in."),
        d("food-05", "food", "Tacos al pastor are marinated pork roasted on a vertical spit."),
        d("food-06", "food", "Sourdough bread relies on wild yeast and a long fermentation."),
        // programming
        d(
            "prog-01",
            "programming",
            "Recursive functions call themselves to solve smaller subproblems."),
        d(
            "prog-02",
            "programming",
            "Java's HashMap uses chained buckets to resolve hash collisions."),
        d(
            "prog-03",
            "programming",
            "Functional programming favors immutable data and pure functions."),
        d("prog-04", "programming", "A binary search runs in logarithmic time on sorted arrays."),
        d(
            "prog-05",
            "programming",
            "Garbage collection reclaims memory occupied by unreachable objects."),
        d("prog-06", "programming", "Concurrent code uses locks or atomics to coordinate threads."),
        // sports
        d("sport-01", "sports", "A marathon is a long-distance running race of 42.195 kilometres."),
        d(
            "sport-02",
            "sports",
            "Tennis grand slams include Wimbledon, Roland Garros, and the US Open."),
        d(
            "sport-03",
            "sports",
            "Soccer's offside rule restricts attackers ahead of the second-to-last defender."),
        d(
            "sport-04",
            "sports",
            "Olympic swimmers train at altitude to boost red blood cell counts."),
        d(
            "sport-05",
            "sports",
            "Cycling time trials reward aerodynamic position and steady pacing."),
        d(
            "sport-06",
            "sports",
            "Rock climbing grades describe the technical difficulty of a route."),
        // music
        d(
            "music-01",
            "music",
            "A symphony orchestra has strings, woodwinds, brass, and percussion."),
        d("music-02", "music", "Jazz improvisation builds melodic ideas over a chord progression."),
        d(
            "music-03",
            "music",
            "Electric guitars use magnetic pickups to convert string vibration to signal."),
        d(
            "music-04",
            "music",
            "Bach's fugues weave a single theme through multiple independent voices."),
        d(
            "music-05",
            "music",
            "Modern pop production layers synthesizers over programmed drum patterns."),
        d(
            "music-06",
            "music",
            "A piano's hammers strike strings to produce its characteristic tone."));
  }

  private static Doc d(String id, String topic, String text) {
    return new Doc(id, topic, text);
  }
}
