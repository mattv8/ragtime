package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Files;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.services.s3.model.*;

class LegacyImporterTest {
    @Test void importsLogicalObjectsAndMetadataWithoutSidecarsOrDeletingOriginals() throws Exception {
        var base = Files.createTempDirectory("legacy-import");
        var root = base.resolve("_object_storage");
        var old = Files.createDirectories(base.resolve("workspaces/ws/s3/buckets/uploads/docs"));
        Files.writeString(old.resolve("plain.txt"), "plain");
        Files.writeString(old.resolve("sdk.txt._S3rver_object"), "sdk");
        Files.writeString(old.resolve("sdk.txt._S3rver_metadata.json"), "{\"content-type\":\"text/plain\",\"x-amz-meta-source\":\"sdk\"}");
        Files.writeString(old.resolve("sdk.txt._S3rver_object.md5"), "internal checksum");
        try (var registry = new Registry(root, "fixture-key"); var engine = new StorageEngine(registry, root, "fixture-key")) {
            registry.mutate(state -> {
                var workspace = state.withObject("workspaces").putObject("ws");
                workspace.put("workspace_id", "ws").put("state", "ready");
                workspace.putArray("buckets").addObject().put("id", "bucket-id").put("name", "uploads").put("backend_id", "local");
            });
            new LegacyImporter(registry, engine).importWorkspace("ws");
            new LegacyImporter(registry, engine).importWorkspace("ws");
            var store = engine.workspaceStore("ws");
            assertEquals(2, store.list(ListObjectsV2Request.builder().bucket("uploads").build()).contents().size());
            try (var response = store.getBlob(GetObjectRequest.builder().bucket("uploads").key("docs/sdk.txt").build())) {
                assertEquals("sdk", new String(response.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
                assertEquals("sdk", response.response().metadata().get("source"));
                assertEquals("text/plain", response.response().contentType());
            }
            assertTrue(Files.exists(old.resolve("sdk.txt._S3rver_object")));
            assertEquals("completed", registry.workspace("ws").path("legacy_import_state").asText());
        }
    }
}
