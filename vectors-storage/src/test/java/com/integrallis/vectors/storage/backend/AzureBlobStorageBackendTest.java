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
package com.integrallis.vectors.storage.backend;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.azure.core.http.HttpResponse;
import com.azure.core.http.rest.PagedIterable;
import com.azure.core.http.rest.Response;
import com.azure.core.util.BinaryData;
import com.azure.storage.blob.BlobClient;
import com.azure.storage.blob.BlobContainerClient;
import com.azure.storage.blob.models.BlobDownloadContentResponse;
import com.azure.storage.blob.models.BlobDownloadHeaders;
import com.azure.storage.blob.models.BlobDownloadResponse;
import com.azure.storage.blob.models.BlobItem;
import com.azure.storage.blob.models.BlobProperties;
import com.azure.storage.blob.models.BlobRange;
import com.azure.storage.blob.models.BlobRequestConditions;
import com.azure.storage.blob.models.BlobStorageException;
import com.azure.storage.blob.models.BlockBlobItem;
import com.azure.storage.blob.models.ListBlobsOptions;
import com.azure.storage.blob.options.BlobParallelUploadOptions;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

/**
 * The backend's own logic over a mocked Azure client: how SDK answers and status codes map onto the
 * {@link StorageBackend} contract. The real wire behaviour is covered by {@link
 * AzureBlobLiveStorageBackendIT} against Azurite or a storage account.
 */
@Tag("unit")
class AzureBlobStorageBackendTest {
  private BlobContainerClient container;
  private BlobClient blob;
  private AzureBlobStorageBackend backend;
  private BlobStorageException notFound;
  private BlobStorageException serverError;
  private BlobStorageException rangeNotSatisfiable;
  private BlobStorageException preconditionFailed;
  private BlobStorageException conflict;

  private static BlobStorageException status(int code) {
    HttpResponse response = mock(HttpResponse.class);
    when(response.getStatusCode()).thenReturn(code);
    return new BlobStorageException("status " + code, response, null);
  }

  @BeforeEach
  void setUp() {
    container = mock(BlobContainerClient.class);
    blob = mock(BlobClient.class);
    when(container.getBlobClient(anyString())).thenReturn(blob);
    backend = new AzureBlobStorageBackend(container);
    notFound = status(404);
    serverError = status(500);
    rangeNotSatisfiable = status(416);
    preconditionFailed = status(412);
    conflict = status(409);
  }

  @Test
  void putOverwritesAndWrapsFailures() throws IOException {
    backend.put("k", new byte[] {1, 2});
    ArgumentCaptor<BinaryData> data = ArgumentCaptor.forClass(BinaryData.class);
    verify(blob).upload(data.capture(), eq(true));
    assertThat(data.getValue().toBytes()).containsExactly(1, 2);

    doAnswer(
            inv -> {
              throw serverError;
            })
        .when(blob)
        .upload(any(BinaryData.class), anyBoolean());
    assertThatThrownBy(() -> backend.put("k", new byte[] {1}))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("put failed");
  }

  @Test
  void getMapsNotFoundToNullAndOtherErrorsToIOException() throws IOException {
    // One consecutive stub: a value, then 404, then 500. Re-stubbing a mock whose current stub
    // throws would run that stub inside when(...).
    when(blob.downloadContent())
        .thenReturn(BinaryData.fromBytes(new byte[] {7}))
        .thenThrow(notFound)
        .thenThrow(serverError);
    assertThat(backend.get("k")).containsExactly(7);
    assertThat(backend.get("k")).isNull();
    assertThatThrownBy(() -> backend.get("k")).isInstanceOf(IOException.class);
  }

  @Test
  void getWithEtagUnquotesTheEtag() throws IOException {
    BlobDownloadContentResponse response = mock(BlobDownloadContentResponse.class);
    BlobDownloadHeaders headers = mock(BlobDownloadHeaders.class);
    when(headers.getETag()).thenReturn("\"0x8DD\"");
    when(response.getValue()).thenReturn(BinaryData.fromBytes(new byte[] {1}));
    when(response.getDeserializedHeaders()).thenReturn(headers);
    when(blob.downloadContentWithResponse(isNull(), isNull(), isNull(), any()))
        .thenReturn(response)
        .thenThrow(notFound)
        .thenThrow(serverError);
    StorageBackend.StoredValue v = backend.getWithEtag("k");
    assertThat(v.value()).containsExactly(1);
    assertThat(v.etag()).isEqualTo("0x8DD");
    assertThat(backend.getWithEtag("k")).isNull();
    assertThatThrownBy(() -> backend.getWithEtag("k")).isInstanceOf(IOException.class);
  }

  @Test
  void openReturnsNullForAMissingBlob() throws IOException {
    when(blob.exists()).thenReturn(false).thenThrow(notFound).thenThrow(serverError);
    assertThat(backend.open("k")).isNull();
    assertThat(backend.open("k")).isNull();
    assertThatThrownBy(() -> backend.open("k")).isInstanceOf(IOException.class);
  }

  @Test
  void getRangeValidatesBoundsBeforeAnyCall() {
    assertThatThrownBy(() -> backend.getRange("k", -1, 1))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> backend.getRange("k", 0, -1))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> backend.getRange("k", Long.MAX_VALUE, 2))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  void getRangeOfLengthZeroChecksExistenceAndOffsetThroughProperties() throws IOException {
    BlobProperties properties = mock(BlobProperties.class);
    when(properties.getBlobSize()).thenReturn(10L);
    when(blob.getProperties())
        .thenReturn(properties)
        .thenReturn(properties)
        .thenThrow(notFound)
        .thenThrow(serverError);
    assertThat(backend.getRange("k", 10, 0)).isEmpty();
    assertThatThrownBy(() -> backend.getRange("k", 11, 0))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThat(backend.getRange("k", 0, 0)).isNull();
    assertThatThrownBy(() -> backend.getRange("k", 0, 0)).isInstanceOf(IOException.class);
  }

  @Test
  void getRangeReturnsExactlyTheRequestedBytesOrFailsPastEof() throws IOException {
    BlobDownloadResponse ok = mock(BlobDownloadResponse.class);
    when(ok.getStatusCode()).thenReturn(206);
    doAnswer(
            inv -> {
              OutputStream out = inv.getArgument(0);
              BlobRange range = inv.getArgument(1);
              assertThat(range.getOffset()).isEqualTo(2L);
              assertThat(range.getCount()).isEqualTo(3L);
              out.write(new byte[] {3, 4, 5});
              return ok;
            })
        .when(blob)
        .downloadStreamWithResponse(
            any(), any(BlobRange.class), isNull(), isNull(), eq(false), isNull(), any());
    assertThat(backend.getRange("k", 2, 3)).containsExactly(3, 4, 5);

    doAnswer(
            inv -> {
              OutputStream out = inv.getArgument(0);
              out.write(new byte[] {9}); // the store clipped the range at the end of the blob
              return ok;
            })
        .when(blob)
        .downloadStreamWithResponse(
            any(), any(BlobRange.class), isNull(), isNull(), eq(false), isNull(), any());
    assertThatThrownBy(() -> backend.getRange("k", 2, 3))
        .isInstanceOf(IndexOutOfBoundsException.class);

    doAnswer(
            inv -> {
              throw rangeNotSatisfiable;
            })
        .when(blob)
        .downloadStreamWithResponse(
            any(), any(BlobRange.class), isNull(), isNull(), eq(false), isNull(), any());
    assertThatThrownBy(() -> backend.getRange("k", 50, 3))
        .isInstanceOf(IndexOutOfBoundsException.class);

    doAnswer(
            inv -> {
              throw notFound;
            })
        .when(blob)
        .downloadStreamWithResponse(
            any(), any(BlobRange.class), isNull(), isNull(), eq(false), isNull(), any());
    assertThat(backend.getRange("k", 0, 3)).isNull();

    doAnswer(
            inv -> {
              throw serverError;
            })
        .when(blob)
        .downloadStreamWithResponse(
            any(), any(BlobRange.class), isNull(), isNull(), eq(false), isNull(), any());
    assertThatThrownBy(() -> backend.getRange("k", 0, 3)).isInstanceOf(IOException.class);
  }

  @Test
  @SuppressWarnings("unchecked")
  void listReturnsBlobNamesUnderThePrefix() throws IOException {
    PagedIterable<BlobItem> page = mock(PagedIterable.class);
    BlobItem a = new BlobItem().setName("p/a");
    BlobItem b = new BlobItem().setName("p/b");
    Iterator<BlobItem> iterator = List.of(a, b).iterator();
    when(page.iterator()).thenReturn(iterator);
    ArgumentCaptor<ListBlobsOptions> options = ArgumentCaptor.forClass(ListBlobsOptions.class);
    when(container.listBlobs(options.capture(), isNull())).thenReturn(page);
    assertThat(backend.list("p/")).containsExactly("p/a", "p/b");
    assertThat(options.getValue().getPrefix()).isEqualTo("p/");

    doThrow(serverError).when(container).listBlobs(any(ListBlobsOptions.class), isNull());
    assertThatThrownBy(() -> backend.list("p/")).isInstanceOf(IOException.class);
  }

  @Test
  void deleteIsIdempotentAndWrapsFailures() throws IOException {
    backend.delete("k");
    verify(blob).deleteIfExists();
    doThrow(serverError).when(blob).deleteIfExists();
    assertThatThrownBy(() -> backend.delete("k")).isInstanceOf(IOException.class);
  }

  @Test
  @SuppressWarnings("unchecked")
  void conditionalPutSendsThePreconditionAndMapsTheOutcome() throws IOException {
    Response<BlockBlobItem> response = mock(Response.class);
    BlockBlobItem item = mock(BlockBlobItem.class);
    when(item.getETag()).thenReturn("\"0xNEW\"");
    when(response.getValue()).thenReturn(item);
    when(blob.uploadWithResponse(any(BlobParallelUploadOptions.class), isNull(), any()))
        .thenReturn(response);

    StorageBackend.ConditionalPutResult created = backend.conditionalPut("k", new byte[] {1}, null);
    assertThat(created.succeeded()).isTrue();
    assertThat(created.newEtag()).isEqualTo("0xNEW");
    backend.conditionalPut("k", new byte[] {2}, "0xOLD");

    ArgumentCaptor<BlobParallelUploadOptions> options =
        ArgumentCaptor.forClass(BlobParallelUploadOptions.class);
    verify(blob, org.mockito.Mockito.times(2))
        .uploadWithResponse(options.capture(), isNull(), any());
    BlobRequestConditions createConditions = options.getAllValues().get(0).getRequestConditions();
    assertThat(createConditions.getIfNoneMatch()).isEqualTo("*");
    assertThat(createConditions.getIfMatch()).isNull();
    BlobRequestConditions updateConditions = options.getAllValues().get(1).getRequestConditions();
    assertThat(updateConditions.getIfMatch()).isEqualTo("\"0xOLD\"");
    assertThat(updateConditions.getIfNoneMatch()).isNull();

    for (BlobStorageException refusal : List.of(preconditionFailed, conflict, notFound)) {
      doThrow(refusal)
          .when(blob)
          .uploadWithResponse(any(BlobParallelUploadOptions.class), isNull(), any());
      StorageBackend.ConditionalPutResult refused =
          backend.conditionalPut("k", new byte[] {3}, "0xOLD");
      assertThat(refused.succeeded()).as(refusal.getMessage()).isFalse();
      assertThat(refused.newEtag()).isNull();
    }
    doThrow(serverError)
        .when(blob)
        .uploadWithResponse(any(BlobParallelUploadOptions.class), isNull(), any());
    assertThatThrownBy(() -> backend.conditionalPut("k", new byte[] {3}, "0xOLD"))
        .isInstanceOf(IOException.class);
  }

  @Test
  void constructorRejectsNull() {
    assertThatThrownBy(() -> new AzureBlobStorageBackend(null))
        .isInstanceOf(NullPointerException.class);
  }
}
