package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Map;
import java.util.NoSuchElementException;
import org.gaul.s3proxy.blobstore.BlobStore;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.services.s3.model.CopyObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadObjectResponse;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;

/** Trusted control-plane rename.  Gateway writes are fenced while copy and delete run. */
final class ObjectRenameService {
  static final class RenameException extends RuntimeException {
    final int status;
    RenameException(int status, String message) { super(message); this.status = status; }
  }

  record Result(long sizeBytes, String contentType) { }

  private final Registry registry;
  private final StorageEngine engine;
  private final Runnable beforeCopy;

  ObjectRenameService(Registry registry, StorageEngine engine) {
    this(registry, engine, () -> { });
  }

  ObjectRenameService(Registry registry, StorageEngine engine, Runnable beforeCopy) {
    this.registry = registry;
    this.engine = engine;
    this.beforeCopy = beforeCopy;
  }

  Result rename(String workspaceId, String bucketName, String key, String newKey) {
    if (key == null || key.isEmpty() || newKey == null || newKey.isEmpty()) {
      throw new RenameException(409, "object key is required");
    }
    if (key.equals(newKey)) throw new RenameException(409, "source and destination keys must differ");
    registry.maintenanceLock().readLock().lock();
    registry.workspaceLock(workspaceId).writeLock().lock();
    try {
      JsonNode workspace = registry.workspace(workspaceId);
      if (workspace == null) throw new RenameException(404, "workspace not found");
      if (!"ready".equals(workspace.path("state").asText("ready"))) {
        throw new RenameException(409, "workspace is fenced");
      }
      JsonNode bucket = bucket(workspace, bucketName);
      if (bucket == null) throw new RenameException(404, "bucket not found");
      String backendId = bucket.path("backend_id").asText();
      String bucketId = bucket.path("id").asText();
      if (backendId.isBlank() || bucketId.isBlank()) throw new RenameException(503, "object storage is unavailable");
      BlobStore store;
      String physicalBucket;
      try {
        store = engine.physicalStore(backendId);
        physicalBucket = engine.physicalBucket(backendId);
      } catch (RuntimeException error) {
        throw new RenameException(503, "object storage is unavailable");
      }
      String prefix = engine.physicalPrefix(workspaceId, bucketId);
      String sourceKey = prefix + key;
      String destinationKey = prefix + newKey;
      if (!store.blobExists(physicalBucket, sourceKey)) throw new RenameException(404, "object not found");
      if (store.blobExists(physicalBucket, destinationKey)) throw new RenameException(409, "destination object already exists");

      HeadObjectResponse source = metadata(store, physicalBucket, sourceKey);
      byte[] sourceHash = sha256(store, physicalBucket, sourceKey);
      beforeCopy.run();
      store.copyBlob(CopyObjectRequest.builder().sourceBucket(physicalBucket).sourceKey(sourceKey)
          .destinationBucket(physicalBucket).destinationKey(destinationKey).build());
      HeadObjectResponse destination = metadata(store, physicalBucket, destinationKey);
      byte[] destinationHash = sha256(store, physicalBucket, destinationKey);
      if (source.contentLength() == null || !source.contentLength().equals(destination.contentLength())
          || !Arrays.equals(sourceHash, destinationHash)
          || !java.util.Objects.equals(source.contentType(), destination.contentType())
          || !sameMetadata(source.metadata(), destination.metadata())) {
        throw new RenameException(503, "object rename verification failed");
      }
      HeadObjectResponse currentSource = metadata(store, physicalBucket, sourceKey);
      if (!source.contentLength().equals(currentSource.contentLength())
          || !Arrays.equals(sourceHash, sha256(store, physicalBucket, sourceKey))) {
        throw new RenameException(409, "source object changed during rename");
      }
      // The workspace write lock excludes every gateway mutation, so this only deletes
      // the source whose bytes were just hashed.  Do not replace this with a blind SDK delete.
      store.removeBlob(DeleteObjectRequest.builder().bucket(physicalBucket).key(sourceKey).build());
      return new Result(destination.contentLength(), destination.contentType());
    } catch (RenameException error) {
      throw error;
    } catch (NoSuchElementException error) {
      throw new RenameException(404, "object not found");
    } catch (RuntimeException error) {
      throw new RenameException(503, "object storage is unavailable");
    } finally {
      registry.workspaceLock(workspaceId).writeLock().unlock();
      registry.maintenanceLock().readLock().unlock();
    }
  }

  private static JsonNode bucket(JsonNode workspace, String name) {
    for (JsonNode bucket : workspace.path("buckets")) if (name.equals(bucket.path("name").asText())) return bucket;
    return null;
  }

  private static HeadObjectResponse metadata(BlobStore store, String bucket, String key) {
    return store.blobMetadata(HeadObjectRequest.builder().bucket(bucket).key(key).build());
  }

  private static byte[] sha256(BlobStore store, String bucket, String key) {
    try (ResponseInputStream<?> stream = store.getBlob(GetObjectRequest.builder().bucket(bucket).key(key).build())) {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      byte[] buffer = new byte[64 * 1024];
      for (int read; (read = stream.read(buffer)) >= 0;) if (read > 0) digest.update(buffer, 0, read);
      return digest.digest();
    } catch (IOException | java.security.GeneralSecurityException error) {
      throw new RenameException(503, "object rename verification failed");
    }
  }

  private static boolean sameMetadata(Map<String, String> source, Map<String, String> destination) {
    return java.util.Objects.equals(source == null ? Map.of() : source, destination == null ? Map.of() : destination);
  }
}
