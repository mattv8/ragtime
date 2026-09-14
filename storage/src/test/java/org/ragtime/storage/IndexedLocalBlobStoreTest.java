package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.services.s3.model.*;

class IndexedLocalBlobStoreTest {
  private static final String BUCKET = "ragtime";

  @Test void preservesUserMetadataAndServesClosedRanges() throws Exception {
    IndexedLocalBlobStore store = new IndexedLocalBlobStore(Files.createTempDirectory("indexed-store"));
    store.putBlob(PutObjectRequest.builder().bucket(BUCKET).key("report.txt")
        .metadata(java.util.Map.of("sha256", "caller-value", "tag", "kept"))
        .contentType("text/plain").cacheControl("no-cache").build(), bytes("abcdef"));

    HeadObjectResponse head = store.blobMetadata(HeadObjectRequest.builder().bucket(BUCKET).key("report.txt").build());
    assertEquals("caller-value", head.metadata().get("sha256"));
    assertEquals("no-cache", head.cacheControl());
    try (var response = store.getBlob(GetObjectRequest.builder().bucket(BUCKET).key("report.txt").range("bytes=2-4").build())) {
      assertEquals(3, response.response().contentLength());
      assertEquals("bytes 2-4/6", response.response().contentRange());
      assertEquals("cde", new String(response.readAllBytes(), StandardCharsets.UTF_8));
    }
  }

  @Test void delimiterListingUsesPrefixRelativeFoldersAndOpaqueCursor() throws Exception {
    IndexedLocalBlobStore store = new IndexedLocalBlobStore(Files.createTempDirectory("indexed-list"));
    for (String key : List.of("namespace/a/one", "namespace/a/two", "namespace/b/one", "namespace/root")) {
      store.putBlob(PutObjectRequest.builder().bucket(BUCKET).key(key).build(), bytes(key));
    }

    ListObjectsV2Response first = store.list(ListObjectsV2Request.builder().bucket(BUCKET).prefix("namespace/").delimiter("/").maxKeys(1).build());
    assertEquals(List.of("namespace/a/"), first.commonPrefixes().stream().map(CommonPrefix::prefix).toList());
    assertTrue(first.isTruncated());
    assertFalse(first.nextContinuationToken().contains("namespace/a/"));
    ListObjectsV2Response second = store.list(ListObjectsV2Request.builder().bucket(BUCKET).prefix("namespace/").delimiter("/").maxKeys(2).continuationToken(first.nextContinuationToken()).build());
    assertEquals(List.of("namespace/b/"), second.commonPrefixes().stream().map(CommonPrefix::prefix).toList());
    assertEquals(List.of("namespace/root"), second.contents().stream().map(S3Object::key).toList());
  }

  @Test void flatPaginationDoesNotSkipKeysExtendingTheLastKey() throws Exception {
    IndexedLocalBlobStore store = new IndexedLocalBlobStore(Files.createTempDirectory("indexed-flat"));
    for (String key : List.of("a", "aa", "ab", "b")) store.putBlob(PutObjectRequest.builder().bucket(BUCKET).key(key).build(), bytes(key));
    String cursor = null;
    java.util.ArrayList<String> keys = new java.util.ArrayList<>();
    do {
      var page = store.list(ListObjectsV2Request.builder().bucket(BUCKET).maxKeys(1).continuationToken(cursor).build());
      keys.addAll(page.contents().stream().map(S3Object::key).toList());
      cursor = page.nextContinuationToken();
      assertTrue(keys.size() <= 4);
    } while (cursor != null);
    assertEquals(List.of("a", "aa", "ab", "b"), keys);
  }

  @Test void multipartHonorsManifestAndRejectsAnotherKey() throws Exception {
    IndexedLocalBlobStore store = new IndexedLocalBlobStore(Files.createTempDirectory("indexed-multipart"));
    var upload = store.initiateMultipartUpload(CreateMultipartUploadRequest.builder().bucket(BUCKET).key("target").metadata(java.util.Map.of("tag", "kept")).build());
    store.uploadMultipartPart(upload, UploadPartRequest.builder().partNumber(1).build(), bytes("unused"));
    var part = store.uploadMultipartPart(upload, UploadPartRequest.builder().partNumber(2).build(), bytes("selected"));
    var alien = new org.gaul.s3proxy.blobstore.domain.MultipartUpload(upload.id(), upload.request().toBuilder().key("alien").build());
    assertEquals("NoSuchUpload", assertThrows(S3Exception.class, () -> store.listMultipartUpload(alien)).awsErrorDetails().errorCode());
    store.completeMultipartUpload(upload, CompleteMultipartUploadRequest.builder().multipartUpload(
        CompletedMultipartUpload.builder().parts(CompletedPart.builder().partNumber(2).eTag(part.eTag()).build()).build()).build());
    try (var result = store.getBlob(GetObjectRequest.builder().bucket(BUCKET).key("target").build())) {
      assertEquals("selected", new String(result.readAllBytes(), StandardCharsets.UTF_8));
      assertEquals("kept", result.response().metadata().get("tag"));
      assertTrue(result.response().eTag().contains("-1"));
    }
    assertTrue(store.listMultipartUploads(BUCKET).isEmpty());
  }

  private static ByteArrayInputStream bytes(String value) {
    return new ByteArrayInputStream(value.getBytes(StandardCharsets.UTF_8));
  }
}
