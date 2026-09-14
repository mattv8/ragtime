package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.lang.reflect.Field;
import java.util.Map;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

class MigrationTest {
  @Test void copiesBytesThenSwitchesBindingWithoutDeletingSource() throws Exception {
    var root=Files.createTempDirectory("migration"); try(var registry=new Registry(root,"key"); var engine=new StorageEngine(registry,root)) {
      installTempTarget(engine,root);
      registry.mutate(s -> { var backends=s.withObject("backends"); var external=backends.putObject("target");external.put("id","target");external.put("type","local"); var w=s.withObject("workspaces").putObject("ws");w.put("workspace_id","ws");w.put("state","ready"); var b=w.putArray("buckets").addObject();b.put("id","bucket-id");b.put("name","bucket");b.put("backend_id","local"); var j=s.withObject("migrations").putObject("job");j.put("id","job");j.put("workspace_id","ws");j.put("target_backend_id","target");j.put("state","pending"); });
      String sourceKey=engine.physicalPrefix("ws","bucket-id")+"nested/value.txt";
      engine.physicalStore("local").putBlob(PutObjectRequest.builder().bucket(engine.physicalBucket("local")).key(sourceKey).build(),new ByteArrayInputStream("source bytes".getBytes(StandardCharsets.UTF_8)));
      new MigrationService(registry,engine).run("job");
      assertEquals("completed",registry.snapshot().path("migrations").path("job").path("state").asText());
      assertEquals("target",registry.workspace("ws").path("buckets").get(0).path("backend_id").asText());
      String destKey=engine.physicalPrefix("ws","bucket-id")+"nested/value.txt";
      assertTrue(engine.physicalStore("local").blobExists(engine.physicalBucket("local"),sourceKey));
      assertTrue(engine.physicalStore("target").blobExists(engine.physicalBucket("target"),destKey));
    }
  }
  @Test void conflictFailsAndLeavesWorkspaceFenced() throws Exception {
    var root=Files.createTempDirectory("migration-conflict"); try(var registry=new Registry(root,"key"); var engine=new StorageEngine(registry,root)) {
      installTempTarget(engine,root);
      registry.mutate(s -> {var backends=s.withObject("backends");var target=backends.putObject("target");target.put("id","target");target.put("type","local");var w=s.withObject("workspaces").putObject("ws");w.put("workspace_id","ws");w.put("state","ready");var b=w.putArray("buckets").addObject();b.put("id","bucket-id");b.put("name","bucket");b.put("backend_id","local");var j=s.withObject("migrations").putObject("job");j.put("id","job");j.put("workspace_id","ws");j.put("target_backend_id","target");j.put("state","pending");});
      String key=engine.physicalPrefix("ws","bucket-id")+"value"; var bytes=new ByteArrayInputStream(new byte[]{1});engine.physicalStore("local").putBlob(PutObjectRequest.builder().bucket(engine.physicalBucket("local")).key(key).build(),bytes);engine.physicalStore("target").putBlob(PutObjectRequest.builder().bucket(engine.physicalBucket("target")).key(key).build(),new ByteArrayInputStream(new byte[]{2}));
      new MigrationService(registry,engine).run("job");
      assertEquals("failed",registry.snapshot().path("migrations").path("job").path("state").asText());assertEquals("migrating",registry.workspace("ws").path("state").asText());assertTrue(engine.physicalStore("local").blobExists(engine.physicalBucket("local"),key));
    }
  }
  @SuppressWarnings("unchecked") private static void installTempTarget(StorageEngine engine,java.nio.file.Path root) throws Exception { Field field=StorageEngine.class.getDeclaredField("physical");field.setAccessible(true);((Map<String,org.gaul.s3proxy.blobstore.BlobStore>)field.get(engine)).put("target",new IndexedLocalBlobStore(root.resolve("temp-target"))); }
}
