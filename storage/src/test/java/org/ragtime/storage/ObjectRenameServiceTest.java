package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

class ObjectRenameServiceTest {
  @Test void pausedRenameBlocksConcurrentGatewayMutationUntilItsDeleteIsSafe() throws Exception {
    try (var registry = new Registry(Files.createTempDirectory("rename"), "key"); var engine = new StorageEngine(registry, Files.createTempDirectory("objects"))) {
      registry.mutate(state -> { var workspace = state.withObject("workspaces").putObject("workspace"); workspace.put("workspace_id", "workspace"); workspace.put("state", "ready"); workspace.put("access_key_id", "key"); workspace.put("secret_access_key", "secret"); workspace.put("default_bucket_name", "default"); var bucket = workspace.withArray("buckets").addObject(); bucket.put("id", "bucket-id"); bucket.put("name", "default"); bucket.put("backend_id", "local"); });
      engine.workspaceStore("workspace").putBlob(PutObjectRequest.builder().bucket("default").key("source").build(), new ByteArrayInputStream("original".getBytes()));
      CountDownLatch copyPaused = new CountDownLatch(1), resumeCopy = new CountDownLatch(1), mutationStarted = new CountDownLatch(1);
      ObjectRenameService service = new ObjectRenameService(registry, engine, () -> { copyPaused.countDown(); try { resumeCopy.await(); } catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new AssertionError(error); } });
      ExecutorService workers = Executors.newFixedThreadPool(2);
      try {
        Future<ObjectRenameService.Result> rename = workers.submit(() -> service.rename("workspace", "default", "source", "renamed"));
        assertTrue(copyPaused.await(5, TimeUnit.SECONDS), "rename did not pause while holding the workspace write lock");
        AtomicReference<Thread> mutationThread = new AtomicReference<>();
        Future<?> mutation = workers.submit(() -> {
          mutationThread.set(Thread.currentThread()); mutationStarted.countDown();
          engine.workspaceStore("workspace").putBlob(PutObjectRequest.builder().bucket("default").key("source").build(), new ByteArrayInputStream("replacement".getBytes()));
        });
        assertTrue(mutationStarted.await(5, TimeUnit.SECONDS), "concurrent gateway writer did not start");
        assertTrue(awaitQueued(registry, mutationThread), "concurrent gateway writer did not queue behind rename workspace write lock");
        assertFalse(mutation.isDone(), "queued gateway mutation completed before rename released its lock");
        resumeCopy.countDown();
        rename.get(10, TimeUnit.SECONDS);
        mutation.get(10, TimeUnit.SECONDS);
        try (var renamed = engine.workspaceStore("workspace").getBlob(GetObjectRequest.builder().bucket("default").key("renamed").build());
            var source = engine.workspaceStore("workspace").getBlob(GetObjectRequest.builder().bucket("default").key("source").build())) {
          assertArrayEquals("original".getBytes(), renamed.readAllBytes());
          assertArrayEquals("replacement".getBytes(), source.readAllBytes());
        }
      } finally {
        resumeCopy.countDown();
        workers.shutdownNow();
        assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS), "rename test workers did not terminate");
      }
    }
  }

  private static boolean awaitQueued(Registry registry, AtomicReference<Thread> thread) throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
    while (System.nanoTime() < deadline) {
      Thread candidate = thread.get();
      if (candidate != null && registry.workspaceLock("workspace").hasQueuedThread(candidate)) return true;
      Thread.sleep(10);
    }
    return false;
  }
}
