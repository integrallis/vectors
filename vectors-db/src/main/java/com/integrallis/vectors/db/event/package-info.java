/**
 * Mutation event model for WAL-based replication and distributed state propagation.
 *
 * <ul>
 *   <li>{@link com.integrallis.vectors.db.event.VectorEvent} — sealed hierarchy (Add, Delete,
 *       Upsert, Commit)
 *   <li>{@link com.integrallis.vectors.db.event.VectorEventCodec} — binary encode/decode
 * </ul>
 */
package com.integrallis.vectors.db.event;
