/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */
package com.integrallis.vectors.ivf;

/** Small test fixture: a document id, the topic it belongs to, and its raw text. */
record Doc(String id, String topic, String text) {}
