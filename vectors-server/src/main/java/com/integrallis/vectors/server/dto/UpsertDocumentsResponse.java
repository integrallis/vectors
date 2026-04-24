package com.integrallis.vectors.server.dto;

/**
 * Outbound body for {@code POST /v1/collections/{name}/documents}.
 *
 * @param upserted number of documents that were applied and committed
 * @param size total live-document count after the commit
 */
public record UpsertDocumentsResponse(int upserted, int size) {}
