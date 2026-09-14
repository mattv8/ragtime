package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import org.gaul.s3proxy.blobstore.BlobStore;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Request;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.S3Object;

/** Copies through physical BlobStores only; source namespaces are never deleted. */
final class MigrationService {
  private static final ObjectMapper JSON=new ObjectMapper();
  private final Registry registry; private final StorageEngine engine;
  MigrationService(Registry registry,StorageEngine engine) { this.registry=registry; this.engine=engine; }

  void run(String jobId) {
    registry.maintenanceLock().readLock().lock();
    try {
      ObjectNode job=job(jobId); String workspaceId=job.path("workspace_id").asText(), target=job.path("target_backend_id").asText();
      try {
      registry.workspaceLock(workspaceId).writeLock().lock();
      try { registry.mutate(s -> { ObjectNode w=workspace(s,workspaceId); w.put("state","migrating"); ObjectNode j=job(s,jobId); j.put("state","copying"); j.remove("error"); }); }
      finally { registry.workspaceLock(workspaceId).writeLock().unlock(); }
      ObjectNode workspace=workspace(workspaceId);
      for (JsonNode bucket : workspace.path("buckets")) copyBucket(jobId,workspaceId,(ObjectNode)bucket,target);
      registry.mutate(s -> job(s,jobId).put("state","verifying"));
      // copyBucket verifies every byte before this state transition; switching is short and atomic.
      registry.workspaceLock(workspaceId).writeLock().lock();
      try { registry.mutate(s -> { ObjectNode w=workspace(s,workspaceId); for(JsonNode b:w.path("buckets")) ((ObjectNode)b).put("backend_id",target); w.put("state","ready"); job(s,jobId).put("state","completed"); }); }
      finally { registry.workspaceLock(workspaceId).writeLock().unlock(); }
      } catch (Exception error) {
        registry.mutate(s -> { ObjectNode j=job(s,jobId); j.put("state","failed"); j.put("error",safe(error)); /* workspace stays migrating: fail closed */ });
      }
    } finally {
      registry.maintenanceLock().readLock().unlock();
    }
  }
  private void copyBucket(String jobId,String workspaceId,ObjectNode bucket,String target) throws Exception {
    String sourceBackend=bucket.path("backend_id").asText("local"), id=bucket.path("id").asText();
    if (sourceBackend.equals(target)) return;
    BlobStore source=engine.physicalStore(sourceBackend), destination=engine.physicalStore(target);
    String sourceBucket=engine.physicalBucket(sourceBackend), destinationBucket=engine.physicalBucket(target);
    String sourcePrefix=engine.physicalPrefix(workspaceId,id), destinationPrefix=engine.physicalPrefix(workspaceId,id);
    String continuation=null;
    do {
      var page=source.list(ListObjectsV2Request.builder().bucket(sourceBucket).prefix(sourcePrefix).continuationToken(continuation).build());
      for(S3Object object:page.contents()) {
        String sourceKey=object.key(), destKey=destinationPrefix+sourceKey.substring(sourcePrefix.length());
        if(destination.blobExists(destinationBucket,destKey)) {
          Digest sourceExisting=digest(source,sourceBucket,sourceKey), destinationExisting=digest(destination,destinationBucket,destKey);
          if(sourceExisting.size!=destinationExisting.size || !MessageDigest.isEqual(sourceExisting.sha,destinationExisting.sha)) throw new IOException("destination conflict for migration object");
          registry.mutate(s -> { ObjectNode j=job(s,jobId); j.put("objects_copied",j.path("objects_copied").asLong()+1); j.put("bytes_copied",j.path("bytes_copied").asLong()+sourceExisting.size); });
          continue;
        }
        Digest sourceDigest=copyAndDigest(source,destination,sourceBucket,destinationBucket,sourceKey,destKey);
        Digest destinationDigest=digest(destination,destinationBucket,destKey);
        if(sourceDigest.size!=destinationDigest.size || !MessageDigest.isEqual(sourceDigest.sha,destinationDigest.sha)) throw new IOException("migration verification mismatch");
        registry.mutate(s -> { ObjectNode j=job(s,jobId); j.put("objects_copied",j.path("objects_copied").asLong()+1); j.put("bytes_copied",j.path("bytes_copied").asLong()+sourceDigest.size); });
      }
      continuation=page.nextContinuationToken();
    } while(continuation!=null && !continuation.isBlank());
  }
  private static Digest copyAndDigest(BlobStore source,BlobStore destination,String sourceBucket,String destinationBucket,String sourceKey,String destKey) throws Exception {
    try(var input=source.getBlob(GetObjectRequest.builder().bucket(sourceBucket).key(sourceKey).build())) {
      CountingDigest stream=new CountingDigest(input);
      var metadata = input.response();
      destination.putBlob(PutObjectRequest.builder().bucket(destinationBucket).key(destKey)
          .contentLength(metadata.contentLength()).contentType(metadata.contentType()).metadata(metadata.metadata())
          .cacheControl(metadata.cacheControl()).contentDisposition(metadata.contentDisposition())
          .contentEncoding(metadata.contentEncoding()).contentLanguage(metadata.contentLanguage()).expires(metadata.expires()).build(),stream);
      return stream.finish();
    }
  }
  private static Digest digest(BlobStore store,String bucket,String key) throws Exception { try(ResponseInputStream<?> input=store.getBlob(GetObjectRequest.builder().bucket(bucket).key(key).build())) { CountingDigest stream=new CountingDigest(input); byte[] buffer=new byte[64*1024]; while(stream.read(buffer)>=0) {} return stream.finish(); } }
  private ObjectNode job(String id) { ObjectNode snapshot=registry.snapshot(); return job(snapshot,id); }
  private static ObjectNode job(ObjectNode state,String id) { JsonNode value=state.path("migrations").get(id); if(!(value instanceof ObjectNode result)) throw new IllegalArgumentException("migration not found"); return result; }
  private ObjectNode workspace(String id) { ObjectNode value=registry.workspace(id); if(value==null) throw new IllegalArgumentException("workspace not found"); return value; }
  private static ObjectNode workspace(ObjectNode state,String id) { JsonNode value=state.path("workspaces").get(id); if(!(value instanceof ObjectNode result)) throw new IllegalArgumentException("workspace not found"); return result; }
  private static String safe(Exception error) { String value=error.getMessage(); return value==null?"migration failed":value.replaceAll("(?i)(secret|password|token)=[^\\s]+","$1=[redacted]"); }
  private record Digest(long size,byte[] sha) {}
  private static final class CountingDigest extends InputStream { private final InputStream source; private final MessageDigest digest; private long count; CountingDigest(InputStream source) throws Exception { this.source=source; digest=MessageDigest.getInstance("SHA-256"); } public int read() throws IOException { int value=source.read(); if(value>=0){digest.update((byte)value);count++;} return value;} public int read(byte[] b,int o,int l)throws IOException{int n=source.read(b,o,l);if(n>0){digest.update(b,o,n);count+=n;}return n;} public void close() throws IOException { source.close(); } Digest finish(){return new Digest(count,digest.digest());} }
}
