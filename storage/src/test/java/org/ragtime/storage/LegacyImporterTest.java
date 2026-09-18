package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Files;
import java.io.RandomAccessFile;
import java.security.MessageDigest;
import java.util.HexFormat;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.services.s3.model.*;

class LegacyImporterTest {
    @Test void importsLogicalObjectsAndMetadataWithoutSidecarsOrDeletingOriginals() throws Exception {
        var base = Files.createTempDirectory("legacy-import");
        var root = base.resolve("_object_storage");
        String generation="0123456789abcdef0123456789abcdef";
        var old = Files.createDirectories(root.resolve("_legacy_imports/ws").resolve(generation).resolve("buckets/uploads/docs"));
        Files.writeString(old.resolve("plain.txt"), "plain");
        Files.writeString(old.resolve("sdk.txt._S3rver_object"), "sdk");
        Files.writeString(old.resolve("sdk.txt._S3rver_metadata.json"), "{\"content-type\":\"text/plain\",\"x-amz-meta-source\":\"sdk\"}");
        Files.writeString(old.resolve("sdk.txt._S3rver_object.md5"), "eae18bc41e1434dd98fa2dd989531da8");
        String plain=sha("plain"),object=sha("sdk"),metadata=sha("{\"content-type\":\"text/plain\",\"x-amz-meta-source\":\"sdk\"}"),md5=sha("eae18bc41e1434dd98fa2dd989531da8");
        String manifest="{\"version\":1,\"workspace_id\":\"ws\",\"generation\":\""+generation+"\",\"files\":[{\"path\":\"buckets/uploads/docs/plain.txt\",\"size\":"+Files.size(old.resolve("plain.txt"))+",\"sha256\":\""+plain+"\"},{\"path\":\"buckets/uploads/docs/sdk.txt._S3rver_metadata.json\",\"size\":"+Files.size(old.resolve("sdk.txt._S3rver_metadata.json"))+",\"sha256\":\""+metadata+"\"},{\"path\":\"buckets/uploads/docs/sdk.txt._S3rver_object\",\"size\":"+Files.size(old.resolve("sdk.txt._S3rver_object"))+",\"sha256\":\""+object+"\"},{\"path\":\"buckets/uploads/docs/sdk.txt._S3rver_object.md5\",\"size\":"+Files.size(old.resolve("sdk.txt._S3rver_object.md5"))+",\"sha256\":\""+md5+"\"}]}";
        String manifestHash=sha(manifest);
        Files.writeString(old.getParent().getParent().getParent().resolve("manifest.json"),manifest);
        try (var registry = new Registry(root, "fixture-key"); var engine = new StorageEngine(registry, root, "fixture-key")) {
            registry.mutate(state -> {
                var workspace = state.withObject("workspaces").putObject("ws");
                workspace.put("workspace_id", "ws").put("state", "importing");
                workspace.putArray("buckets").addObject().put("id", "bucket-id").put("name", "uploads").put("backend_id", "local");
                state.withObject("legacy_import_jobs").putObject("ws").put("workspace_id","ws").put("generation",generation).put("manifest_sha256",manifestHash).put("state","copying");
            });
            new LegacyImporter(registry, engine).importStaged("ws",generation,manifestHash);
            var store = engine.workspaceStore("ws");
            assertEquals(2, store.list(ListObjectsV2Request.builder().bucket("uploads").build()).contents().size());
            try (var response = store.getBlob(GetObjectRequest.builder().bucket("uploads").key("docs/sdk.txt").build())) {
                assertEquals("sdk", new String(response.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
                assertEquals("sdk", response.response().metadata().get("source"));
                assertEquals("text/plain", response.response().contentType());
            }
            assertTrue(Files.exists(old.resolve("sdk.txt._S3rver_object")));
        }
    }
    @Test void rejectsUnlistedOrMissingStagedArtifactsWithoutCopying() throws Exception {
        var root=Files.createTempDirectory("legacy-import"); String generation="0123456789abcdef0123456789abcdef";
        var bucket=Files.createDirectories(root.resolve("_legacy_imports/ws").resolve(generation).resolve("buckets/uploads")); Files.writeString(bucket.resolve("listed.txt"),"listed"); Files.writeString(bucket.resolve("unlisted.txt"),"unlisted");
        String listed=sha("listed"), manifest="{\"version\":1,\"workspace_id\":\"ws\",\"generation\":\""+generation+"\",\"files\":[{\"path\":\"buckets/uploads/listed.txt\",\"size\":6,\"sha256\":\""+listed+"\"}]}"; Files.writeString(bucket.getParent().getParent().resolve("manifest.json"),manifest);
        try(var registry=new Registry(root,"fixture-key"); var engine=new StorageEngine(registry,root,"fixture-key")) {
            registry.mutate(state -> { var workspace=state.withObject("workspaces").putObject("ws"); workspace.put("workspace_id","ws").put("state","ready"); workspace.putArray("buckets").addObject().put("id","bucket-id").put("name","uploads").put("backend_id","local"); });
            assertThrows(java.io.IOException.class,()->new LegacyImporter(registry,engine).importStaged("ws",generation,sha(manifest)));
        }
    }
    @Test void acceptsAPreviouslyBoundUnsortedNestedManifest() throws Exception {
        var root=Files.createTempDirectory("legacy-import"); String generation="0123456789abcdef0123456789abcdef";
        var bucket=Files.createDirectories(root.resolve("_legacy_imports/ws").resolve(generation).resolve("buckets/uploads/dir"));
        Files.writeString(bucket.resolve("file.txt"), "nested"); Files.writeString(bucket.getParent().resolve("dir.txt"), "sibling");
        String manifest=manifest(generation,
                "buckets/uploads/dir/file.txt", "nested",
                "buckets/uploads/dir.txt", "sibling");
        Files.writeString(bucket.getParent().getParent().getParent().resolve("manifest.json"),manifest);
        try(var registry=new Registry(root,"fixture-key"); var engine=new StorageEngine(registry,root,"fixture-key")) {
            importingWorkspace(registry, generation, sha(manifest));
            new LegacyImporter(registry,engine).importStaged("ws",generation,sha(manifest));
            assertEquals(2, engine.workspaceStore("ws").list(ListObjectsV2Request.builder().bucket("uploads").build()).contents().size());
        }
    }
    @Test void acceptsPythonCodePointOrderedSupplementaryUnicodeManifest() throws Exception {
        var root=Files.createTempDirectory("legacy-import"); String generation="0123456789abcdef0123456789abcdef";
        var bucket=Files.createDirectories(root.resolve("_legacy_imports/ws").resolve(generation).resolve("buckets/uploads"));
        String privateUse="\uE000.txt", supplementary="\uD800\uDC00.txt";
        Files.writeString(bucket.resolve(privateUse), "private"); Files.writeString(bucket.resolve(supplementary), "supplementary");
        String manifest=manifest(generation,
                "buckets/uploads/" + privateUse, "private",
                "buckets/uploads/" + supplementary, "supplementary");
        Files.writeString(bucket.getParent().getParent().resolve("manifest.json"),manifest);
        try(var registry=new Registry(root,"fixture-key"); var engine=new StorageEngine(registry,root,"fixture-key")) {
            importingWorkspace(registry, generation, sha(manifest));
            new LegacyImporter(registry,engine).importStaged("ws",generation,sha(manifest));
            assertEquals(2, engine.workspaceStore("ws").list(ListObjectsV2Request.builder().bucket("uploads").build()).contents().size());
        }
    }
    @Test void rejectsOversizedManifestBeforeReadingItsBody() throws Exception {
        var root=Files.createTempDirectory("legacy-import"); String generation="0123456789abcdef0123456789abcdef";
        var manifest=root.resolve("_legacy_imports/ws").resolve(generation).resolve("manifest.json"); Files.createDirectories(manifest.getParent());
        try (var file = new RandomAccessFile(manifest.toFile(), "rw")) { file.setLength((long) LegacyImporter.MAX_MANIFEST_OR_EVIDENCE_BYTES + 1); }
        try(var registry=new Registry(root,"fixture-key"); var engine=new StorageEngine(registry,root,"fixture-key")) {
            assertThrows(java.io.IOException.class,()->new LegacyImporter(registry,engine).importStaged("ws",generation,"a".repeat(64)));
        }
    }
    private static void importingWorkspace(Registry registry, String generation, String manifestHash) {
        registry.mutate(state -> { var workspace=state.withObject("workspaces").putObject("ws"); workspace.put("workspace_id","ws").put("state","importing"); workspace.putArray("buckets").addObject().put("id","bucket-id").put("name","uploads").put("backend_id","local"); state.withObject("legacy_import_jobs").putObject("ws").put("workspace_id","ws").put("generation",generation).put("manifest_sha256",manifestHash).put("state","copying"); });
    }
    private static String manifest(String generation, String firstPath, String firstContents, String secondPath, String secondContents) throws Exception {
        return "{\"version\":1,\"workspace_id\":\"ws\",\"generation\":\""+generation+"\",\"files\":[{\"path\":\""+firstPath+"\",\"size\":"+firstContents.getBytes().length+",\"sha256\":\""+sha(firstContents)+"\"},{\"path\":\""+secondPath+"\",\"size\":"+secondContents.getBytes().length+",\"sha256\":\""+sha(secondContents)+"\"}]}";
    }
    private static String sha(String value) throws Exception { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value.getBytes())); }
}
