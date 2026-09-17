package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.gaul.s3proxy.blobstore.BlobStore;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

/** Imports only an immutable Ragtime-published staging generation. */
final class LegacyImporter {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String OBJECT_SUFFIX = "._S3rver_object";
    private final Registry registry;
    private final StorageEngine engine;

    record Result(String evidenceSha256, List<String> verifiedFiles) { }
    LegacyImporter(Registry registry, StorageEngine engine) { this.registry = registry; this.engine = engine; }

    Result importStaged(String workspaceId, String generation, String manifestSha256) throws Exception {
        Path generationRoot = registry.root().resolve("_legacy_imports").resolve(workspaceId).resolve(generation);
        Path manifestPath = generationRoot.resolve("manifest.json");
        requireNoSymlinkAncestors(generationRoot); if (Files.isSymbolicLink(manifestPath)) throw new IOException("symlink in staged import");
        byte[] manifestBytes = Files.readAllBytes(manifestPath);
        if (!sha256(manifestBytes).equals(manifestSha256)) throw new IOException("staged manifest digest mismatch");
        JsonNode manifest = JSON.readTree(manifestBytes);
        validateManifest(manifest, workspaceId, generation);
        Path buckets = generationRoot.resolve("buckets");
        if (!Files.isDirectory(buckets, LinkOption.NOFOLLOW_LINKS) || Files.isSymbolicLink(generationRoot) || Files.isSymbolicLink(buckets)) throw new IOException("invalid staged buckets directory");
        Map<String, Entry> entries = validateFiles(generationRoot, manifest.path("files"));
        ObjectNode workspace = registry.workspace(workspaceId);
        if (workspace == null || "revoked".equals(workspace.path("state").asText())) throw new IOException("workspace unavailable");
        List<String> verified = new ArrayList<>();
        for (Entry entry : entries.values()) {
            String relative = entry.path;
            if (relative.endsWith("._S3rver_metadata.json") || relative.endsWith("._S3rver_object.md5") || relative.contains("/._S3rver_")) continue;
            Path file = generationRoot.resolve(relative);
            String keyPath = relative.substring("buckets/".length());
            int slash = keyPath.indexOf('/');
            if (slash < 1) throw new IOException("staged object has no bucket");
            String bucketName = keyPath.substring(0, slash), key = keyPath.substring(slash + 1);
            JsonNode bucket = findBucket(workspace, bucketName);
            if (bucket == null) throw new IOException("unconfigured staged bucket");
            boolean s3rver = key.endsWith(OBJECT_SUFFIX);
            if (!s3rver && entries.containsKey(relative + OBJECT_SUFFIX)) throw new IOException("conflicting plain and S3rver object representations");
            if (s3rver && entries.containsKey(relative.substring(0, relative.length() - OBJECT_SUFFIX.length()))) throw new IOException("conflicting plain and S3rver object representations");
            if (s3rver) {
                String logical = key.substring(0, key.length() - OBJECT_SUFFIX.length());
                String metadataPath = relative.substring(0, relative.length() - OBJECT_SUFFIX.length()) + "._S3rver_metadata.json";
                JsonNode headers = JSON.createObjectNode();
                if (entries.containsKey(metadataPath)) {
                    if (Files.size(generationRoot.resolve(metadataPath)) > 1024 * 1024) throw new IOException("S3rver metadata too large");
                    headers = JSON.readTree(Files.readAllBytes(generationRoot.resolve(metadataPath)));
                    if (headers == null || !headers.isObject()) throw new IOException("invalid S3rver metadata");
                }
                copy(workspaceId, generation, manifestSha256, bucket, logical, file, headers, entry);
                verified.add(relative);
                if (entries.containsKey(metadataPath)) verified.add(metadataPath);
                String md5Path = relative + ".md5";
                if (entries.containsKey(md5Path) && md5Matches(file, generationRoot.resolve(md5Path))) verified.add(md5Path);
            } else {
                copy(workspaceId, generation, manifestSha256, bucket, key, file, JSON.createObjectNode(), entry);
                verified.add(relative);
            }
        }
        verified.sort(String::compareTo);
        ObjectNode evidence = JSON.createObjectNode().put("version", 1).put("workspace_id", workspaceId).put("generation", generation).put("manifest_sha256", manifestSha256);
        ArrayNode array = evidence.putArray("verified_files"); verified.forEach(array::add);
        byte[] evidenceBytes = JSON.writeValueAsBytes(evidence);
        Path temporary = generationRoot.resolve("verification.json.tmp");
        Files.write(temporary, evidenceBytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        try (FileChannel channel=FileChannel.open(temporary, StandardOpenOption.WRITE)) { channel.force(true); }
        Files.move(temporary, generationRoot.resolve("verification.json"), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        try (FileChannel channel=FileChannel.open(generationRoot, StandardOpenOption.READ)) { channel.force(true); }
        return new Result(sha256(evidenceBytes), List.copyOf(verified));
    }

    private static void validateManifest(JsonNode node, String workspaceId, String generation) throws IOException {
        if (!node.isObject() || node.path("version").asInt() != 1 || !workspaceId.equals(node.path("workspace_id").asText()) || !generation.equals(node.path("generation").asText()) || !node.path("files").isArray()) throw new IOException("invalid staged manifest");
    }
    private static Map<String, Entry> validateFiles(Path root, JsonNode files) throws Exception {
        Map<String, Entry> result = new HashMap<>(); String previous = "";
        for (JsonNode item : files) {
            String path = item.path("path").asText(); long size = item.path("size").asLong(-1); String digest = item.path("sha256").asText();
            if (!path.matches("buckets/(?:[^/]+/)*[^/]+") || path.contains("//") || path.contains("/../") || path.contains("/./") || path.endsWith("/..") || path.endsWith("/.") || path.compareTo(previous) <= 0 || size < 0 || !digest.matches("[0-9a-f]{64}") || result.putIfAbsent(path, new Entry(path, size, digest)) != null) throw new IOException("invalid staged manifest file");
            previous = path;
        }
        Set<String> actual = new HashSet<>();
        try (var paths = Files.walk(root.resolve("buckets"))) {
            for (Path path : (Iterable<Path>) paths::iterator) {
                if (Files.isSymbolicLink(path)) throw new IOException("symlink in staged import");
                if (Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) actual.add("buckets/" + root.resolve("buckets").relativize(path).toString().replace(path.getFileSystem().getSeparator(), "/"));
                else if (!Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) throw new IOException("nonregular staged entry");
            }
        }
        if (!actual.equals(result.keySet())) throw new IOException("staged manifest does not match files");
        for (Entry entry : result.values()) {
            Path path = root.resolve(entry.path);
            if (Files.size(path) != entry.size || !HexFormat.of().formatHex(hash(Files.newInputStream(path))).equals(entry.sha256)) throw new IOException("staged file digest mismatch");
        }
        return result;
    }
    private static boolean md5Matches(Path object, Path sidecar) throws Exception {
        if (Files.size(sidecar)>1024) return false;
        String expected = Files.readString(sidecar, StandardCharsets.UTF_8).trim();
        if (!expected.matches("(?i)[0-9a-f]{32}")) return false;
        try (InputStream input = Files.newInputStream(object)) {
            MessageDigest digest = MessageDigest.getInstance("MD5"); byte[] buffer = new byte[65536];
            for (int read; (read = input.read(buffer)) >= 0;) if (read > 0) digest.update(buffer, 0, read);
            return expected.equalsIgnoreCase(HexFormat.of().formatHex(digest.digest()));
        }
    }
    private void copy(String workspaceId, String generation, String manifestSha256, JsonNode bucket, String key, Path source, JsonNode headers, Entry entry) throws Exception {
        if (headers.toString().getBytes(StandardCharsets.UTF_8).length > 1024 * 1024) throw new IOException("S3rver metadata too large");
        copyVerified(workspaceId,generation,manifestSha256,bucket,key,source,headers,entry);
    }
    private void copyVerified(String workspaceId, String generation, String manifestSha256, JsonNode bucket, String key, Path source, JsonNode headers, Entry entry) throws Exception {
        registry.maintenanceLock().readLock().lock();
        registry.workspaceLock(workspaceId).readLock().lock();
        try {
            ObjectNode job=LegacyImportService.job(registry.snapshot(),workspaceId); ObjectNode workspace=registry.workspace(workspaceId);
            if(job==null || workspace==null || !generation.equals(job.path("generation").asText()) || !manifestSha256.equals(job.path("manifest_sha256").asText()) || !LegacyImportService.active(job) || !"importing".equals(workspace.path("state").asText())) throw new IOException("legacy import is no longer eligible");
            BlobStore store = engine.physicalStore(bucket.path("backend_id").asText());
            String physicalBucket = engine.physicalBucket(bucket.path("backend_id").asText()), destination = engine.physicalPrefix(workspaceId, bucket.path("id").asText()) + key;
            Map<String, String> userMetadata = new HashMap<>();
            headers.fields().forEachRemaining(header -> { if (header.getKey().startsWith("x-amz-meta-")) userMetadata.put(header.getKey().substring(11), header.getValue().asText()); });
            byte[] expected = HexFormat.of().parseHex(entry.sha256);
            if (Files.size(source)!=entry.size || !MessageDigest.isEqual(expected,hash(Files.newInputStream(source)))) throw new IOException("staged source changed during import");
            String contentType=headers.path("content-type").asText(Files.probeContentType(Path.of(key))), cacheControl=headers.path("cache-control").asText(null), contentDisposition=headers.path("content-disposition").asText(null), contentEncoding=headers.path("content-encoding").asText(null), contentLanguage=headers.path("content-language").asText(null);
            if (!store.blobExists(physicalBucket, destination)) try (InputStream input = Files.newInputStream(source)) {
                store.putBlob(PutObjectRequest.builder().bucket(physicalBucket).key(destination).contentLength(entry.size).contentType(contentType).metadata(userMetadata).cacheControl(cacheControl).contentDisposition(contentDisposition).contentEncoding(contentEncoding).contentLanguage(contentLanguage).build(), input);
            }
            try (var result = store.getBlob(GetObjectRequest.builder().bucket(physicalBucket).key(destination).build())) {
                if (result.response().contentLength() != entry.size || !MessageDigest.isEqual(expected, hash(result)) || !userMetadata.equals(result.response().metadata()) || !Objects.equals(contentType,result.response().contentType()) || !Objects.equals(cacheControl,result.response().cacheControl()) || !Objects.equals(contentDisposition,result.response().contentDisposition()) || !Objects.equals(contentEncoding,result.response().contentEncoding()) || !Objects.equals(contentLanguage,result.response().contentLanguage())) throw new IOException("legacy import conflict or verification mismatch");
            }
        } finally { registry.workspaceLock(workspaceId).readLock().unlock(); registry.maintenanceLock().readLock().unlock(); }
    }
    private void requireNoSymlinkAncestors(Path path) throws IOException {
        Path root=registry.root().toAbsolutePath().normalize(), current=path.toAbsolutePath().normalize();
        if(!current.startsWith(root)) throw new IOException("staged path escapes storage root");
        for(Path part=root; !part.equals(current); part=part.resolve(current.subpath(part.getNameCount(),part.getNameCount()+1))) if(Files.isSymbolicLink(part)) throw new IOException("symlink in staged import");
        if(Files.isSymbolicLink(current)) throw new IOException("symlink in staged import");
    }
    private static JsonNode findBucket(ObjectNode workspace, String name) { for (JsonNode bucket : workspace.path("buckets")) if (name.equals(bucket.path("name").asText())) return bucket; return null; }
    private static byte[] hash(InputStream source) throws Exception { try (source) { MessageDigest digest = MessageDigest.getInstance("SHA-256"); byte[] buffer = new byte[65536]; for (int read; (read = source.read(buffer)) >= 0;) if (read > 0) digest.update(buffer, 0, read); return digest.digest(); } }
    static String sha256(byte[] source) throws Exception { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(source)); }
    private record Entry(String path, long size, String sha256) { }
}
