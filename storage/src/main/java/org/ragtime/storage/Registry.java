package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;
import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;

/** Durable encrypted gateway metadata.  Its immutable cache is the S3 hot path. */
public final class Registry implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final byte[] CONTEXT = "ragtime-object-storage-state-v1\0".getBytes(StandardCharsets.UTF_8);
    private final Path database;
    private final Path root;
    private final byte[] aesKey;
    private final SecureRandom random = new SecureRandom();
    private final ReentrantReadWriteLock stateLock = new ReentrantReadWriteLock();
    private final ReentrantReadWriteLock maintenanceLock = new ReentrantReadWriteLock(true);
    private final ConcurrentHashMap<String, ReentrantReadWriteLock> workspaceLocks = new ConcurrentHashMap<>();
    private volatile State state;

    private record State(ObjectNode snapshot, Map<String, String> identities) { }

    public Registry(Path root, String encryptionKey) throws IOException {
        String normalized = encryptionKey == null ? "" : encryptionKey.strip();
        if (normalized.isEmpty()) throw new IOException("managed encryption key is unavailable");
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(CONTEXT);
            this.aesKey = digest.digest(normalized.getBytes(StandardCharsets.UTF_8));
        } catch (GeneralSecurityException error) { throw new IOException("cannot derive registry key", error); }
        Files.createDirectories(root);
        this.root = root;
        database = root.resolve("registry.sqlite");
        this.state = loadOrCreate();
    }

    public ObjectNode snapshot() { return state.snapshot.deepCopy(); }
    public String installationId() { return state.snapshot.path("installation_id").asText(); }
    public ObjectNode workspace(String id) {
        JsonNode value = state.snapshot.path("workspaces").get(id);
        return value instanceof ObjectNode node ? node.deepCopy() : null;
    }
    public ObjectNode findIdentity(String accessKey) {
        String workspaceId = state.identities.get(accessKey);
        return workspaceId == null ? null : workspace(workspaceId);
    }
    public ObjectNode backend(String id) {
        JsonNode value = state.snapshot.path("backends").get(id);
        return value instanceof ObjectNode node ? node.deepCopy() : null;
    }
    public ReentrantReadWriteLock workspaceLock(String id) {
        return workspaceLocks.computeIfAbsent(id, ignored -> new ReentrantReadWriteLock(true));
    }
    public ReentrantReadWriteLock maintenanceLock() { return maintenanceLock; }
    Path root() { return root; }

    /** Persists before publishing, so a crash can never expose uncommitted state. */
    public void mutate(Consumer<ObjectNode> change) {
        maintenanceLock.readLock().lock();
        stateLock.writeLock().lock();
        try {
            ObjectNode next = state.snapshot.deepCopy();
            change.accept(next);
            Map<String, String> identities = identityIndex(next);
            persist(next);
            state = new State(next, Map.copyOf(identities));
        } finally {
            stateLock.writeLock().unlock();
            maintenanceLock.readLock().unlock();
        }
    }

    /** Flushes WAL while callers hold the maintenance fence. */
    public void checkpoint() throws IOException {
        try (Connection connection = connection(); Statement statement = connection.createStatement()) {
            statement.execute("PRAGMA wal_checkpoint(FULL)");
        } catch (SQLException error) { throw new IOException("registry checkpoint failed", error); }
    }

    private State loadOrCreate() throws IOException {
        try (Connection connection = connection(); Statement statement = connection.createStatement()) {
            statement.execute("PRAGMA journal_mode=WAL");
            statement.execute("CREATE TABLE IF NOT EXISTS registry_state (id INTEGER PRIMARY KEY CHECK(id=1), payload BLOB NOT NULL)");
            try (ResultSet rows = statement.executeQuery("SELECT payload FROM registry_state WHERE id=1")) {
                if (!rows.next()) {
                    ObjectNode initial = initialState(); persist(initial); return new State(initial, Map.of());
                }
                ObjectNode loaded = (ObjectNode) JSON.readTree(decrypt(rows.getBytes(1)));
                if (loaded == null || loaded.path("version").asInt() != 1) throw new IOException("invalid registry state");
                return new State(loaded, Map.copyOf(identityIndex(loaded)));
            }
        } catch (SQLException | GeneralSecurityException error) {
            throw new IOException("encrypted registry cannot be opened", error);
        }
    }

    private Connection connection() throws SQLException { return DriverManager.getConnection("jdbc:sqlite:" + database); }
    private void persist(ObjectNode value) {
        try (Connection connection = connection(); PreparedStatement statement = connection.prepareStatement(
                "INSERT INTO registry_state(id,payload) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload")) {
            connection.setAutoCommit(false);
            statement.setBytes(1, encrypt(JSON.writeValueAsBytes(value)));
            statement.executeUpdate(); connection.commit();
        } catch (SQLException | IOException | GeneralSecurityException error) {
            throw new IllegalStateException("registry persistence failed", error);
        }
    }
    private byte[] encrypt(byte[] plain) throws GeneralSecurityException {
        byte[] nonce = new byte[12]; random.nextBytes(nonce);
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding"); cipher.init(Cipher.ENCRYPT_MODE, new javax.crypto.spec.SecretKeySpec(aesKey, "AES"), new GCMParameterSpec(128, nonce));
        byte[] cipherText = cipher.doFinal(plain), result = new byte[nonce.length + cipherText.length];
        System.arraycopy(nonce, 0, result, 0, nonce.length); System.arraycopy(cipherText, 0, result, nonce.length, cipherText.length); return result;
    }
    private byte[] decrypt(byte[] payload) throws GeneralSecurityException, IOException {
        if (payload == null || payload.length <= 12) throw new IOException("invalid encrypted registry payload");
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding"); cipher.init(Cipher.DECRYPT_MODE, new javax.crypto.spec.SecretKeySpec(aesKey, "AES"), new GCMParameterSpec(128, payload, 0, 12));
        return cipher.doFinal(payload, 12, payload.length - 12);
    }
    private static Map<String, String> identityIndex(ObjectNode value) {
        Map<String, String> result = new HashMap<>();
        Iterator<Map.Entry<String, JsonNode>> fields = value.path("workspaces").fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> entry = fields.next(); JsonNode workspace = entry.getValue();
            if ("revoked".equals(workspace.path("state").asText())) continue;
            String key = workspace.path("access_key_id").asText();
            if (key.isBlank()) continue;
            if (result.putIfAbsent(key, entry.getKey()) != null) throw new IllegalArgumentException("duplicate object storage identity");
        }
        return result;
    }
    private static ObjectNode initialState() {
        ObjectNode result = JSON.createObjectNode(); result.put("version", 1); result.put("installation_id", UUID.randomUUID().toString()); result.put("default_backend_id", "local");
        ObjectNode backend = result.putObject("backends").putObject("local"); backend.put("id", "local"); backend.put("type", "local");
        result.putObject("workspaces"); result.putObject("migrations"); result.putObject("legacy_import_jobs"); return result;
    }
    @Override public void close() { }
}
