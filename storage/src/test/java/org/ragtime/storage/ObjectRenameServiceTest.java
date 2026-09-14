package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

class ObjectRenameServiceTest {
  @Test void pausedRenameBlocksConcurrentGatewayMutationUntilItsDeleteIsSafe() throws Exception {
    try (var registry = new Registry(Files.createTempDirectory("rename"), "key"); var engine = new StorageEngine(registry, Files.createTempDirectory("objects"))) {
      registry.mutate(state -> { var workspace = state.withObject("workspaces").putObject("workspace"); workspace.put("workspace_id", "workspace"); workspace.put("state", "ready"); workspace.put("access_key_id", "key"); workspace.put("secret_access_key", "secret"); workspace.put("default_bucket_name", "default"); var bucket = workspace.withArray("buckets").addObject(); bucket.put("id", "bucket-id"); bucket.put("name", "default"); bucket.put("backend_id", "local"); });
      engine.workspaceStore("workspace").putBlob(PutObjectRequest.builder().bucket("default").key("source").build(), new ByteArrayInputStream("original".getBytes()));
      CountDownLatch copyPaused = new CountDownLatch(1), resumeCopy = new CountDownLatch(1), mutationDone = new CountDownLatch(1);
      ObjectRenameService service = new ObjectRenameService(registry, engine, () -> { copyPaused.countDown(); try { resumeCopy.await(); } catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new AssertionError(error); } });
      Thread rename = new Thread(() -> service.rename("workspace", "default", "source", "renamed")); rename.start();
      assertTrue(copyPaused.await(1, TimeUnit.SECONDS));
      Thread concurrentMutation = new Thread(() -> { engine.workspaceStore("workspace").putBlob(PutObjectRequest.builder().bucket("default").key("source").build(), new ByteArrayInputStream("replacement".getBytes())); mutationDone.countDown(); }); concurrentMutation.start();
      assertFalse(mutationDone.await(100, TimeUnit.MILLISECONDS));
      resumeCopy.countDown(); rename.join(1000); assertFalse(rename.isAlive()); assertTrue(mutationDone.await(1, TimeUnit.SECONDS));
      assertTrue(engine.workspaceStore("workspace").blobExists("default", "renamed"));
    }
  }
}
