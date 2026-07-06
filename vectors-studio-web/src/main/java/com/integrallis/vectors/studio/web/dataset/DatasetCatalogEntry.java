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
package com.integrallis.vectors.studio.web.dataset;

/**
 * One entry in the sample-dataset catalog (a row of {@code sample-datasets.json}). Describes a
 * HuggingFace datasets-server source and how to map its columns into a Studio collection. The entry
 * {@code id} doubles as the target collection name.
 *
 * @param id stable identifier; also the collection name once loaded
 * @param name human-readable title shown in the datasets table
 * @param description one-line summary
 * @param domain corpus domain (e.g. Wikipedia, arXiv)
 * @param model embedding model that produced the vectors
 * @param dimension vector dimensionality
 * @param hfDataset HuggingFace dataset id passed to the datasets-server {@code /rows} API
 * @param config HF dataset config name (usually {@code default})
 * @param split HF split name (usually {@code train})
 * @param vectorColumn column holding the embedding array
 * @param textColumn column holding raw text, or {@code null}
 * @param idColumn column holding the document id, or {@code null}
 * @param metric similarity function name (COSINE/EUCLIDEAN/DOT_PRODUCT/MAXIMUM_INNER_PRODUCT)
 * @param defaultLimit default number of rows to load
 */
public record DatasetCatalogEntry(
    String id,
    String name,
    String description,
    String domain,
    String model,
    int dimension,
    String hfDataset,
    String config,
    String split,
    String vectorColumn,
    String textColumn,
    String idColumn,
    String metric,
    int defaultLimit) {}
