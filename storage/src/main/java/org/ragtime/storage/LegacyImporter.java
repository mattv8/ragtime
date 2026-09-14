package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Map;
import org.gaul.s3proxy.blobstore.BlobStore;
import software.amazon.awssdk.services.s3.model.*;

/** Idempotent import of completed plain/S3rver objects; originals are never modified. */
final class LegacyImporter {
    private static final String OBJECT_SUFFIX = "._S3rver_object";
    private final Registry registry;
    private final StorageEngine engine;

    LegacyImporter(Registry registry, StorageEngine engine) { this.registry = registry; this.engine = engine; }

    void importWorkspace(String id) throws Exception {
        if (!id.matches("[A-Za-z0-9][A-Za-z0-9_-]{0,127}")) throw new IOException("Invalid workspace ID");
        registry.maintenanceLock().readLock().lock();
        try {
            registry.workspaceLock(id).writeLock().lock();
            try {
                ObjectNode current = registry.workspace(id);
                if (current == null || "revoked".equals(current.path("state").asText())) throw new IOException("Workspace unavailable");
                if ("completed".equals(current.path("legacy_import_state").asText())) return;
                registry.mutate(state -> ((ObjectNode) state.path("workspaces").get(id)).put("state", "importing").put("legacy_import_state", "copying"));
            } finally { registry.workspaceLock(id).writeLock().unlock(); }

            Path root = registry.root().getParent().resolve("workspaces").resolve(id).resolve("s3").resolve("buckets");
            ObjectNode workspace = registry.workspace(id);
            if (Files.exists(root, LinkOption.NOFOLLOW_LINKS)) {
                Path realRoot = root.toRealPath();
                if (!realRoot.equals(root.toAbsolutePath().normalize())) throw new IOException("Symlinked legacy root");
                try (var paths = Files.walk(root)) {
                    for (Path file : (Iterable<Path>) paths::iterator) {
                        if (Files.isSymbolicLink(file)) throw new IOException("Symlink in legacy storage");
                        if (!Files.isRegularFile(file, LinkOption.NOFOLLOW_LINKS)) continue;
                        Path relative = root.relativize(file);
                        if (relative.getNameCount() < 2 || !file.toRealPath().startsWith(realRoot)) throw new IOException("Invalid legacy object path");
                        String bucketName = relative.getName(0).toString();
                        JsonNode bucket = null;
                        for (JsonNode candidate : workspace.path("buckets")) if (bucketName.equals(candidate.path("name").asText())) bucket = candidate;
                        if (bucket == null) throw new IOException("Unconfigured legacy bucket");
                        String key = relative.subpath(1, relative.getNameCount()).toString().replace(file.getFileSystem().getSeparator(), "/");
                        if (key.startsWith("._S3rver_")) continue; // bucket config/incomplete old multipart uploads remain in originals
                        if (key.endsWith("._S3rver_metadata.json") || key.endsWith("._S3rver_object.md5")) continue;
                        boolean s3rver = key.endsWith(OBJECT_SUFFIX);
                        String logicalKey = s3rver ? key.substring(0, key.length() - OBJECT_SUFFIX.length()) : key;
                        Path competing = s3rver ? root.resolve(bucketName).resolve(logicalKey) : file.resolveSibling(file.getFileName() + OBJECT_SUFFIX);
                        if (Files.isRegularFile(competing, LinkOption.NOFOLLOW_LINKS)) throw new IOException("Conflicting plain and S3rver object representations");
                        JsonNode headers = new ObjectMapper().createObjectNode();
                        if (s3rver) {
                            Path sidecar = file.resolveSibling(file.getFileName().toString().replaceFirst("\\._S3rver_object$", "._S3rver_metadata.json"));
                            if (Files.exists(sidecar, LinkOption.NOFOLLOW_LINKS)) {
                                if (Files.isSymbolicLink(sidecar) || Files.size(sidecar) > 1024 * 1024) throw new IOException("Invalid S3rver metadata");
                                headers = new ObjectMapper().readTree(Files.readString(sidecar));
                            }
                        }
                        copy(id, bucket, logicalKey, file, headers);
                    }
                }
            }
            registry.workspaceLock(id).writeLock().lock();
            try { registry.mutate(state -> ((ObjectNode) state.path("workspaces").get(id)).put("state", "ready").put("legacy_import_state", "completed")); }
            finally { registry.workspaceLock(id).writeLock().unlock(); }
        } finally { registry.maintenanceLock().readLock().unlock(); }
    }

    private void copy(String workspaceId, JsonNode bucket, String key, Path source, JsonNode headers) throws Exception {
        String backend = bucket.path("backend_id").asText();
        BlobStore store = engine.physicalStore(backend);
        String physicalBucket = engine.physicalBucket(backend), destination = engine.physicalPrefix(workspaceId, bucket.path("id").asText()) + key;
        Map<String, String> userMetadata = new HashMap<>();
        headers.fields().forEachRemaining(entry -> { if (entry.getKey().startsWith("x-amz-meta-")) userMetadata.put(entry.getKey().substring(11), entry.getValue().asText()); });
        String contentType = headers.path("content-type").asText(Files.probeContentType(Path.of(key)));
        byte[] expected;
        try (InputStream input = Files.newInputStream(source)) { expected = hash(input); }
        if (!store.blobExists(physicalBucket, destination)) {
            try (InputStream input = Files.newInputStream(source)) {
                store.putBlob(PutObjectRequest.builder().bucket(physicalBucket).key(destination).contentLength(Files.size(source))
                    .contentType(contentType).metadata(userMetadata).cacheControl(headers.path("cache-control").asText(null))
                    .contentDisposition(headers.path("content-disposition").asText(null)).contentEncoding(headers.path("content-encoding").asText(null))
                    .contentLanguage(headers.path("content-language").asText(null)).build(), input);
            }
        }
        try (var result = store.getBlob(GetObjectRequest.builder().bucket(physicalBucket).key(destination).build())) {
            if (result.response().contentLength() != Files.size(source) || !MessageDigest.isEqual(expected, hash(result))
                || !userMetadata.equals(result.response().metadata())) throw new IOException("Legacy import conflict or verification mismatch");
        }
    }

    private static byte[] hash(InputStream source) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[65536];
        for (int read; (read = source.read(buffer)) >= 0;) if (read > 0) digest.update(buffer, 0, read);
        return digest.digest();
    }
}
