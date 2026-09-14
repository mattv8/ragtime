package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import org.gaul.s3proxy.blobstore.BlobStore;
import org.gaul.s3proxy.awssdk.AwsS3SdkBlobStore;
import org.gaul.s3proxy.blobstore.Credentials;

/** Backend selection and namespace construction. Registry remains the authority for bindings. */
public final class StorageEngine implements AutoCloseable {
  private final Registry registry; private final Path root; private final String encryptionKey; private final Map<String,BlobStore> physical=new ConcurrentHashMap<>(); private final Map<String,BlobStore> workspaces=new ConcurrentHashMap<>();
  public StorageEngine(Registry registry, Path root) { this(registry, root, ""); }
  StorageEngine(Registry registry, Path root, String encryptionKey) { this.registry=registry; this.root=root; this.encryptionKey=encryptionKey; }
  Registry registry() { return registry; }
  public BlobStore workspaceStore(String workspaceId) { return workspaces.computeIfAbsent(workspaceId, id -> new NamespaceBlobStore(this,id,encryptionKey,physicalStore("local"))); }
  public BlobStore physicalStore(String backendId) { return physical.computeIfAbsent(backendId, this::openPhysical); }
  private BlobStore openPhysical(String id) {
    JsonNode backend=registry.backend(id);
    if (backend == null) throw new IllegalStateException("unknown storage backend");
    if ("local".equals(backend.path("type").asText()))
      return new IndexedLocalBlobStore(root.resolve("backends").resolve(id));
     if ("external".equals(backend.path("type").asText()) || "s3".equals(backend.path("type").asText())) return new RepeatableS3Store(new AwsS3SdkBlobStore(
      () -> new Credentials(backend.path("access_key_id").asText(), backend.path("secret_access_key").asText()),
       backend.path("endpoint").asText(), backend.path("region").asText("us-east-1"), "native", "false", "false"));
    throw new IllegalStateException("unsupported storage backend");
  }
  public String physicalBucket(String backendId) { JsonNode b=registry.backend(backendId); if(b==null) throw new IllegalStateException("unknown storage backend"); return "local".equals(b.path("type").asText())?"ragtime":b.path("bucket").asText(); }
  public String physicalPrefix(String workspaceId,String bucketId) { return registry.installationId()+"/"+workspaceId+"/"+bucketId+"/"; }
  void requireWritable(String workspaceId) { JsonNode w=registry.workspace(workspaceId); if(w==null || !"ready".equals(w.path("state").asText("ready"))) throw new IllegalStateException("workspace is fenced"); }
  <T> T mutate(String workspaceId, Supplier<T> operation) {
    registry.maintenanceLock().readLock().lock(); registry.workspaceLock(workspaceId).readLock().lock();
    try { requireWritable(workspaceId); return operation.get(); }
    finally { registry.workspaceLock(workspaceId).readLock().unlock(); registry.maintenanceLock().readLock().unlock(); }
  }
  public void checkpoint() { physical.values().forEach(store -> { if (store instanceof IndexedLocalBlobStore indexed) indexed.checkpoint(); }); }
  @Override public void close() { physical.values().forEach(BlobStore::close); physical.clear(); workspaces.clear(); }
}
