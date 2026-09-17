package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.gaul.s3proxy.blobstore.BlobStore;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import java.io.ByteArrayInputStream;

/** Private, bearer-protected control endpoint.  It deliberately has no provider credentials in responses. */
public final class ControlServer implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final SecureRandom RANDOM = new SecureRandom();
    private final Registry registry;
    private final StorageEngine engine;
    private final InetSocketAddress address;
    private final byte[] token;
    private HttpServer server;
    private ExecutorService executor;
    private final ExecutorService migrationExecutor = Executors.newSingleThreadExecutor(r -> new Thread(r, "object-storage-migration"));
    private final MigrationService migrationService;
    private final LegacyImportService legacyImports;
    private final ObjectRenameService renameService;
    private final AtomicReference<BackupLease> backup = new AtomicReference<>();

    private static final class BackupLease {
        final String id = UUID.randomUUID().toString(); final Object monitor = new Object(); volatile boolean release;
        volatile Exception failure; final Thread owner;
        final CountDownLatch acquired = new CountDownLatch(1);
        BackupLease(Registry registry, StorageEngine engine) {
            owner = new Thread(() -> {
                registry.maintenanceLock().writeLock().lock();
                try { registry.checkpoint(); engine.checkpoint(); acquired.countDown(); synchronized (monitor) { while (!release) monitor.wait(); } }
                catch (Exception error) { failure = error; }
                finally { acquired.countDown(); registry.maintenanceLock().writeLock().unlock(); }
            }, "object-storage-backup-fence");
        }
    }

    public ControlServer(Registry registry, StorageEngine engine, InetSocketAddress address, String token) {
        this.registry = registry; this.engine = engine; this.address = address;
        this.token = token == null ? new byte[0] : token.strip().getBytes(StandardCharsets.UTF_8);
        this.migrationService = new MigrationService(registry, engine);
        this.legacyImports = new LegacyImportService(registry, engine);
        this.renameService = new ObjectRenameService(registry, engine);
        registry.snapshot().path("migrations").elements().forEachRemaining(job -> {
            String state = job.path("state").asText();
            if (state.equals("pending") || state.equals("copying") || state.equals("verifying"))
                migrationExecutor.submit(() -> migrationService.run(job.path("id").asText()));
        });
        registry.snapshot().path("legacy_import_jobs").fieldNames().forEachRemaining(workspaceId -> {
            ObjectNode job = LegacyImportService.job(registry.snapshot(), workspaceId);
            if (job != null && LegacyImportService.active(job)) migrationExecutor.submit(() -> legacyImports.run(workspaceId));
        });
    }
    public void start() throws IOException {
        server = HttpServer.create(address, 32); executor = Executors.newFixedThreadPool(8, r -> new Thread(r, "object-storage-control")); server.setExecutor(executor);
        server.createContext("/", this::handle); server.start();
    }
    @Override public void close() {
        BackupLease lease = backup.getAndSet(null); if (lease != null) release(lease);
        if (server != null) server.stop(0); if (executor != null) executor.shutdownNow(); migrationExecutor.shutdownNow();
    }

    private void handle(HttpExchange exchange) throws IOException {
        try {
            String path = exchange.getRequestURI().getPath();
            if ("/health".equals(path)) { respond(exchange, 200, JSON.createObjectNode().put("status", "ready")); return; }
            if (!authorized(exchange)) { error(exchange, 401, "unauthorized"); return; }
            ObjectNode body = readBody(exchange);
            ObjectNode result = dispatch(exchange.getRequestMethod(), path, body);
            respond(exchange, 200, result);
        } catch (BadRequest error) { error(exchange, error.status, error.getMessage()); }
        catch (ObjectRenameService.RenameException error) { error(exchange, error.status, error.getMessage()); }
        catch (IllegalArgumentException error) { error(exchange, 409, error.getMessage()); }
        catch (Exception error) { error(exchange, 500, "object storage control request failed"); }
        finally { exchange.close(); }
    }
    private boolean authorized(HttpExchange exchange) {
        String auth = exchange.getRequestHeaders().getFirst("Authorization");
        if (auth == null || !auth.startsWith("Bearer ")) return false;
        return MessageDigest.isEqual(token, auth.substring(7).strip().getBytes(StandardCharsets.UTF_8));
    }
    private ObjectNode dispatch(String method, String path, ObjectNode body) {
        if ("GET".equals(method) && "/v1/settings".equals(path)) return settings();
        if ("PUT".equals(method) && "/v1/settings".equals(path)) return putSettings(body);
        if ("POST".equals(method) && "/v1/settings/test".equals(path)) return testSettings(body);
        if ("GET".equals(method) && "/v1/migrations".equals(path)) return migrations();
        if ("POST".equals(method) && "/v1/migrations".equals(path)) return createMigrations(body);
        if ("POST".equals(method) && "/v1/backup/prepare".equals(path)) return prepareBackup();
        if ("POST".equals(method) && "/v1/backup/release".equals(path)) return releaseBackup(body);
        String[] parts = path.split("/");
        if (parts.length == 5 && "v1".equals(parts[1]) && "migrations".equals(parts[2]) && "retry".equals(parts[4]) && "POST".equals(method)) return retryMigration(parts[3]);
        if (parts.length >= 4 && "v1".equals(parts[1]) && "workspaces".equals(parts[2])) return workspaceRoute(method, parts, body);
        throw new BadRequest(404, "not found");
    }
    private ObjectNode workspaceRoute(String method, String[] parts, ObjectNode body) {
        String workspaceId = parts[3]; validWorkspace(workspaceId);
        if (parts.length == 4) {
            if ("GET".equals(method)) return requireWorkspace(workspaceId);
            if ("DELETE".equals(method)) return deleteWorkspace(workspaceId);
        }
        if (parts.length == 5 && "ensure".equals(parts[4]) && "POST".equals(method)) return ensure(workspaceId, body);
        if (parts.length == 5 && "legacy-import".equals(parts[4])) {
            if ("POST".equals(method)) return importLegacy(workspaceId, body);
            if ("GET".equals(method)) return legacyImportStatus(workspaceId);
        }
        if (parts.length == 6 && "legacy-import".equals(parts[4]) && "gc".equals(parts[5]) && "POST".equals(method)) return acknowledgeLegacyGc(workspaceId, body);
        if (parts.length == 5 && "import-legacy".equals(parts[4]) && "POST".equals(method)) return retiredLegacyImport(workspaceId);
        if (parts.length == 5 && "buckets".equals(parts[4]) && "POST".equals(method)) return createBucket(workspaceId, body);
        if (parts.length == 6 && "buckets".equals(parts[4])) {
            if ("PUT".equals(method)) return updateBucket(workspaceId, parts[5], body);
            if ("DELETE".equals(method)) return deleteBucket(workspaceId, parts[5]);
        }
        if (parts.length == 7 && "buckets".equals(parts[4]) && "rename".equals(parts[6]) && "POST".equals(method)) return renameObject(workspaceId, parts[5], body);
        throw new BadRequest(404, "not found");
    }
    private ObjectNode renameObject(String workspaceId, String bucket, ObjectNode body) {
        ObjectRenameService.Result result = renameService.rename(workspaceId, bucket, body.path("key").asText(), body.path("new_key").asText());
        return JSON.createObjectNode().put("workspace_id", workspaceId).put("bucket_name", bucket)
            .put("key", body.path("new_key").asText()).put("size_bytes", result.sizeBytes())
            .put("content_type", result.contentType());
    }
    private ObjectNode settings() {
        ObjectNode state = registry.snapshot(), backend = registry.backend(state.path("default_backend_id").asText()); ObjectNode result = JSON.createObjectNode();
        result.put("mode", backend != null && "s3".equals(backend.path("type").asText()) ? "external" : "local"); result.put("default_backend_id", state.path("default_backend_id").asText("local"));
        if (backend != null) { copyPublic(backend, result, "endpoint", "region", "bucket"); result.put("access_key_configured", !backend.path("access_key_id").asText().isBlank()); result.put("secret_key_configured", !backend.path("secret_access_key").asText().isBlank()); }
        int local=0; for(JsonNode workspace:state.path("workspaces")) for(JsonNode bucket:workspace.path("buckets")) if("local".equals(bucket.path("backend_id").asText())) {local++;break;} result.put("existing_local_workspaces",local); result.set("migrations", migrations().path("jobs")); return result;
    }
    private ObjectNode putSettings(ObjectNode body) {
        String mode = body.path("mode").asText("local"); if (!mode.equals("local") && !mode.equals("external")) throw new BadRequest(400, "invalid storage mode");
        if (mode.equals("external")) { required(body, "endpoint"); required(body, "region"); required(body, "bucket"); }
        if(mode.equals("external")) {
            String id="external-"+UUID.randomUUID(); addExternalBackend(id,body); try { probeBackend(id,body.path("create_bucket").asBoolean(false)); } catch(Exception error) { registry.mutate(state -> state.withObject("backends").remove(id)); throw new BadRequest(400,"external storage validation failed"); } registry.mutate(state -> state.put("default_backend_id",id)); return settings();
        }
        registry.mutate(state -> {
            if (mode.equals("local")) { state.put("default_backend_id", "local"); return; }
        }); return settings();
    }
    private ObjectNode testSettings(ObjectNode body) {
        required(body, "endpoint"); required(body, "region"); required(body, "bucket");
        String id="probe-"+UUID.randomUUID(); addExternalBackend(id,body); try { probeBackend(id,body.path("create_bucket").asBoolean(false)); return JSON.createObjectNode().put("success",true); }
        catch(Exception error) { throw new BadRequest(400,"external storage probe failed"); }
        finally { registry.mutate(state -> state.withObject("backends").remove(id)); }
    }
    private void addExternalBackend(String id,ObjectNode body) { registry.mutate(state -> { ObjectNode previous=registry.backend(state.path("default_backend_id").asText()); ObjectNode backend=state.withObject("backends").putObject(id);backend.put("id",id);backend.put("type","s3");copyPresent(body,backend,"endpoint","region","bucket"); for(String credential:new String[]{"access_key_id","secret_access_key"}) { String value=body.path(credential).asText(); if(!value.isBlank()) backend.put(credential,value); else if(previous!=null && "s3".equals(previous.path("type").asText())) backend.put(credential,previous.path(credential).asText()); } }); }
    private void probeBackend(String id, boolean createBucket) {
        BlobStore store=engine.physicalStore(id); String bucket=engine.physicalBucket(id), key="__ragtime_control_probe__/"+UUID.randomUUID();
        try { store.headBucket(software.amazon.awssdk.services.s3.model.HeadBucketRequest.builder().bucket(bucket).build()); }
        catch(Exception missing) { if(!createBucket) throw new IllegalStateException("configured root bucket is unavailable",missing); store.createContainer(software.amazon.awssdk.services.s3.model.CreateBucketRequest.builder().bucket(bucket).build()); }
        store.putBlob(PutObjectRequest.builder().bucket(bucket).key(key).contentLength(1L).build(),new ByteArrayInputStream(new byte[]{1}));
        try(var ignored=store.getBlob(GetObjectRequest.builder().bucket(bucket).key(key).build())) { if(ignored.read()!=1) throw new IllegalStateException("probe verification failed"); }
        catch(Exception error) { throw new IllegalStateException("probe verification failed",error); }
        finally { try { store.removeBlob(DeleteObjectRequest.builder().bucket(bucket).key(key).build()); } catch(Exception ignored) {} }
    }
    private ObjectNode ensure(String id, ObjectNode legacy) {
        registry.mutate(state -> {
            if (state.path("workspaces").has(id)) return;
            ObjectNode workspace = state.withObject("workspaces").putObject(id); workspace.put("workspace_id", id); workspace.put("state", "ready");
            if (legacy.has("buckets")) workspace.put("legacy_import_state", "pending");
            String requested = legacy.path("access_key_id").asText(); boolean collision = !requested.isBlank() && identityUsed(state,requested,id); String key = uniqueAccessKey(state, requested.isBlank() ? "ragtime-" + randomHex(12) : requested, id);
            workspace.put("access_key_id", key); workspace.put("secret_access_key", legacy.path("secret_access_key").asText(randomHex(32)));
            if(collision) workspace.put("identity_rekeyed",true);
            String name = bucketName(legacy.path("default_bucket_name").asText("default")); workspace.put("default_bucket_name", name);
            if(legacy.path("buckets").isArray()) for(JsonNode raw:legacy.path("buckets")) { String bucket=bucketName(raw.path("name").asText()); if(findBucket(workspace,bucket)==null) addBucket(workspace,bucket,state.path("default_backend_id").asText(),bucket.equals(name),raw); }
            if(findBucket(workspace,name)==null) addBucket(workspace,name,state.path("default_backend_id").asText(),true,JSON.createObjectNode());
            syncDefaults(workspace);
        }); return requireWorkspace(id);
    }
    private ObjectNode createBucket(String id, ObjectNode body) {
        String name = bucketName(body.path("name").asText()); registry.mutate(state -> { ObjectNode ws = requiredWorkspace(state, id); requireReady(ws); if (findBucket(ws, name) != null) throw new IllegalArgumentException("bucket already exists"); addBucket(ws, name, state.path("default_backend_id").asText(), body.path("make_default").asBoolean(false), body); syncDefaults(ws); }); return requireWorkspace(id);
    }
    private ObjectNode updateBucket(String id, String oldName, ObjectNode body) {
        registry.mutate(state -> { ObjectNode ws = requiredWorkspace(state, id); requireReady(ws); ObjectNode bucket = findBucket(ws, oldName); if (bucket == null) throw new BadRequest(404,"bucket not found"); String renamed = body.has("new_name") ? bucketName(body.path("new_name").asText()) : oldName; ObjectNode other = findBucket(ws, renamed); if (other != null && other != bucket) throw new IllegalArgumentException("bucket already exists"); bucket.put("name", renamed); bucket.put("updated_at",Instant.now().toString()); if (oldName.equals(ws.path("default_bucket_name").asText()) || body.path("make_default").asBoolean(false)) ws.put("default_bucket_name", renamed); syncDefaults(ws); }); return requireWorkspace(id);
    }
    private ObjectNode deleteBucket(String id, String name) { registry.mutate(state -> { ObjectNode ws = requiredWorkspace(state, id); requireReady(ws); ArrayNode buckets = (ArrayNode) ws.path("buckets"); if (buckets.size() <= 1) throw new IllegalArgumentException("workspace requires one bucket"); for (int i=0;i<buckets.size();i++) if (name.equals(buckets.get(i).path("name").asText())) { buckets.remove(i); if (name.equals(ws.path("default_bucket_name").asText())) ws.put("default_bucket_name", buckets.get(0).path("name").asText()); syncDefaults(ws); return; } throw new BadRequest(404,"bucket not found"); }); return JSON.createObjectNode().put("success", true).put("workspace_id", id).put("bucket_name", name); }
    private ObjectNode deleteWorkspace(String id) {
        registry.mutate(state -> { ObjectNode workspace=requiredWorkspace(state,id); workspace.put("state","revoked"); workspace.put("deletion_state","retained_tombstone"); workspace.put("deleted_at",Instant.now().toString()); });
        return JSON.createObjectNode().put("success",true).put("workspace_id",id).put("cleanup_state","retained_tombstone");
    }
    private ObjectNode importLegacy(String id, ObjectNode body) {
        String generation = body.path("generation").asText(), manifestSha = body.path("manifest_sha256").asText();
        if (!generation.matches("[0-9a-f]{32}") || !manifestSha.matches("[0-9a-f]{64}")) throw new BadRequest(400, "invalid staged import identity");
        final boolean[] enqueue={false};
        registry.maintenanceLock().readLock().lock(); registry.workspaceLock(id).writeLock().lock();
        try { registry.mutate(state -> {
            ObjectNode workspace=requiredWorkspace(state,id); if("revoked".equals(workspace.path("state").asText())) throw new BadRequest(404,"workspace not found");
            ObjectNode existing=LegacyImportService.job(state,id);
            if(existing!=null) { if(!generation.equals(existing.path("generation").asText()) || !manifestSha.equals(existing.path("manifest_sha256").asText())) throw new BadRequest(409,"legacy import identity conflict"); if("failed".equals(existing.path("state").asText())) { existing.put("state","pending").remove("error"); enqueue[0]=true; } return; }
            if("completed".equals(workspace.path("legacy_import_state").asText()) || !"pending".equals(workspace.path("legacy_import_state").asText()) || !"ready".equals(workspace.path("state").asText())) throw new BadRequest(409,"workspace is not eligible for staged legacy import");
            ObjectNode job=state.withObject("legacy_import_jobs").putObject(id); job.put("workspace_id",id).put("generation",generation).put("manifest_sha256",manifestSha).put("state","pending").put("gc_completed",false); workspace.put("state","importing"); enqueue[0]=true;
        }); } finally { registry.workspaceLock(id).writeLock().unlock(); registry.maintenanceLock().readLock().unlock(); }
        if(enqueue[0]) migrationExecutor.submit(() -> legacyImports.run(id)); return legacyImportStatus(id);
    }
    private ObjectNode retiredLegacyImport(String id) {
        ObjectNode job = LegacyImportService.job(registry.snapshot(), id);
        if (job != null && "completed".equals(job.path("state").asText())) return legacyImportStatus(id);
        throw new BadRequest(409, "legacy import requires a staged manifest");
    }
    private ObjectNode legacyImportStatus(String id) {
        ObjectNode job = LegacyImportService.job(registry.snapshot(), id); if (job == null) throw new BadRequest(404, "legacy import not found");
        ObjectNode result=JSON.createObjectNode(); for(String field:new String[]{"workspace_id","generation","manifest_sha256","state"}) result.put(field,job.path(field).asText()); result.put("gc_completed",job.path("gc_completed").asBoolean(false));
        if ("completed".equals(job.path("state").asText()) && !job.path("gc_completed").asBoolean(false)) {
            try { Path evidence=registry.root().resolve("_legacy_imports").resolve(id).resolve(job.path("generation").asText()).resolve("verification.json"); byte[] bytes=Files.readAllBytes(evidence); if(!LegacyImporter.sha256(bytes).equals(job.path("verification_sha256").asText())) throw new IOException("verification evidence mismatch"); JsonNode node=JSON.readTree(bytes); if(!id.equals(node.path("workspace_id").asText()) || !job.path("generation").asText().equals(node.path("generation").asText()) || !job.path("manifest_sha256").asText().equals(node.path("manifest_sha256").asText()) || !node.path("verified_files").isArray()) throw new IOException("invalid verification evidence"); result.set("verified_files",node.path("verified_files")); }
            catch(Exception error) { result.put("verification_evidence_available",false); }
        }
        return result;
    }
    private ObjectNode acknowledgeLegacyGc(String id, ObjectNode body) {
        String generation=body.path("generation").asText(), manifestSha=body.path("manifest_sha256").asText(); ObjectNode job=LegacyImportService.job(registry.snapshot(),id);
        if(job==null || !"completed".equals(job.path("state").asText()) || !generation.equals(job.path("generation").asText()) || !manifestSha.equals(job.path("manifest_sha256").asText())) throw new BadRequest(409,"legacy import identity conflict");
        registry.mutate(state -> LegacyImportService.job(state,id).put("gc_completed",true)); return legacyImportStatus(id);
    }
    private ObjectNode createMigrations(ObjectNode body) {
        ArrayNode jobs = JSON.createArrayNode(); List<String> ids = new ArrayList<>(); ObjectNode snapshot=registry.snapshot();
        if (body.path("workspace_ids").isArray()) body.path("workspace_ids").forEach(n -> ids.add(n.asText())); else snapshot.path("workspaces").fieldNames().forEachRemaining(ids::add);
        String target=body.path("target_backend_id").asText(snapshot.path("default_backend_id").asText()); if(registry.backend(target)==null) throw new BadRequest(400,"target backend not found");
        registry.mutate(state -> { for(String id:ids) { ObjectNode workspace=requiredWorkspace(state,id); if(!"ready".equals(workspace.path("state").asText())) continue; boolean needs=false; for(JsonNode bucket:workspace.path("buckets")) if(!target.equals(bucket.path("backend_id").asText())) {needs=true;break;} if(!needs || activeJob(state,id)) continue; String jobId=UUID.randomUUID().toString(); ObjectNode job=state.withObject("migrations").putObject(jobId);job.put("id",jobId);job.put("workspace_id",id);job.put("target_backend_id",target);job.put("state","pending");job.put("objects_copied",0);job.put("bytes_copied",0);job.put("created_at",Instant.now().toString());jobs.add(job.deepCopy()); }});
        for(JsonNode job:jobs) migrationExecutor.submit(() -> migrationService.run(job.path("id").asText())); return JSON.createObjectNode().set("jobs",jobs);
    }
    private ObjectNode migrations() { ArrayNode jobs = JSON.createArrayNode(); registry.snapshot().path("migrations").elements().forEachRemaining(n -> jobs.add(n.deepCopy())); return JSON.createObjectNode().set("jobs", jobs); }
    private static boolean activeJob(ObjectNode state,String workspaceId) { for(JsonNode job:state.path("migrations")) if(workspaceId.equals(job.path("workspace_id").asText()) && (job.path("state").asText().equals("pending")||job.path("state").asText().equals("copying")||job.path("state").asText().equals("verifying"))) return true; return false; }
    private ObjectNode retryMigration(String id) { registry.mutate(state -> { JsonNode node = state.path("migrations").get(id); if (!(node instanceof ObjectNode job)) throw new IllegalArgumentException("migration not found"); String status = job.path("state").asText(); if (!status.equals("failed") && !status.equals("pending")) throw new IllegalArgumentException("migration cannot be retried"); job.put("state","pending"); job.remove("error"); }); migrationExecutor.submit(() -> migrationService.run(id)); return migrations(); }
    private ObjectNode prepareBackup() {
        BackupLease lease = new BackupLease(registry, engine); if (!backup.compareAndSet(null, lease)) throw new BadRequest(409, "backup is already prepared"); lease.owner.start();
        try { if (!lease.acquired.await(5, TimeUnit.SECONDS)) { backup.compareAndSet(lease, null); release(lease); throw new BadRequest(503, "backup preparation timed out"); } }
        catch (InterruptedException error) { Thread.currentThread().interrupt(); backup.compareAndSet(lease, null); release(lease); throw new BadRequest(503,"backup preparation interrupted"); }
        if (lease.failure != null) { backup.compareAndSet(lease, null); throw new BadRequest(503, "backup preparation failed"); }
        return JSON.createObjectNode().put("consistent", true).put("lease_id", lease.id);
    }
    private ObjectNode releaseBackup(ObjectNode body) { BackupLease lease = backup.get(); if (lease == null || !lease.id.equals(body.path("lease_id").asText())) throw new BadRequest(404, "backup lease not found"); backup.compareAndSet(lease, null); release(lease); return JSON.createObjectNode().put("released", true); }
    private static void release(BackupLease lease) { synchronized (lease.monitor) { lease.release = true; lease.monitor.notifyAll(); } }
    private ObjectNode requireWorkspace(String id) { ObjectNode value = registry.workspace(id); if (value == null) throw new BadRequest(404, "workspace not found"); return value; }
    private static ObjectNode requiredWorkspace(ObjectNode state, String id) { JsonNode value = state.path("workspaces").get(id); if (!(value instanceof ObjectNode node)) throw new IllegalArgumentException("workspace not found"); return node; }
    private static void requireReady(ObjectNode workspace) { if(!"ready".equals(workspace.path("state").asText("ready"))) throw new BadRequest(409,"workspace is fenced"); }
    private static ObjectNode findBucket(ObjectNode workspace, String name) { for (JsonNode bucket : workspace.path("buckets")) if (name.equals(bucket.path("name").asText())) return (ObjectNode) bucket; return null; }
    private static void addBucket(ObjectNode workspace, String name, String backend, boolean makeDefault, JsonNode source) { ObjectNode bucket = workspace.withArray("buckets").addObject(); bucket.put("id", UUID.randomUUID().toString()); bucket.put("name", name); bucket.put("backend_id", backend); copyPresent(source, bucket, "description", "public_prefix", "private_prefix"); bucket.put("created_at",Instant.now().toString()); bucket.put("updated_at",Instant.now().toString()); bucket.put("is_default", makeDefault); if (makeDefault) workspace.put("default_bucket_name", name); }
    private static void syncDefaults(ObjectNode workspace) { String current=workspace.path("default_bucket_name").asText(); for(JsonNode value:workspace.path("buckets")) ((ObjectNode)value).put("is_default",current.equals(value.path("name").asText())); }
    private static String uniqueAccessKey(ObjectNode state, String requested, String workspace) { for (JsonNode existing : state.path("workspaces")) if (!workspace.equals(existing.path("workspace_id").asText()) && requested.equals(existing.path("access_key_id").asText())) return "ragtime-" + randomHex(16); return requested; }
    private static boolean identityUsed(ObjectNode state,String requested,String workspace) { for(JsonNode existing:state.path("workspaces")) if(!workspace.equals(existing.path("workspace_id").asText()) && requested.equals(existing.path("access_key_id").asText())) return true; return false; }
    private static String bucketName(String name) { if (!name.matches("[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")) throw new BadRequest(400, "invalid bucket name"); return name; }
    private static void validWorkspace(String value) { if (!value.matches("[A-Za-z0-9][A-Za-z0-9_-]{0,127}")) throw new BadRequest(400, "invalid workspace id"); }
    private static void required(ObjectNode node, String field) { if (node.path(field).asText().isBlank()) throw new BadRequest(400, "missing " + field); }
    private static void copyPresent(JsonNode from, ObjectNode to, String... names) { for (String name : names) if (from.has(name) && !from.path(name).isNull()) to.put(name, from.path(name).asText()); }
    private static void copyPublic(ObjectNode from, ObjectNode to, String... names) { for (String name : names) if (from.has(name)) to.put(name, from.path(name).asText()); }
    private static String randomHex(int bytes) { byte[] value = new byte[bytes]; RANDOM.nextBytes(value); StringBuilder result = new StringBuilder(bytes*2); for (byte b:value) result.append(String.format("%02x", b)); return result.toString(); }
    private static ObjectNode readBody(HttpExchange exchange) throws IOException { long length=exchange.getRequestHeaders().getFirst("Content-Length")==null?-1:Long.parseLong(exchange.getRequestHeaders().getFirst("Content-Length")); if(length>1_048_576) throw new BadRequest(413,"request body is too large"); byte[] data=exchange.getRequestBody().readNBytes(1_048_577); if(data.length>1_048_576) throw new BadRequest(413,"request body is too large"); return data.length == 0 ? JSON.createObjectNode() : (ObjectNode) JSON.readTree(data); }
    private static void respond(HttpExchange exchange, int status, ObjectNode value) throws IOException { byte[] data = JSON.writeValueAsBytes(value); exchange.getResponseHeaders().set("Content-Type", "application/json"); exchange.sendResponseHeaders(status, data.length); exchange.getResponseBody().write(data); }
    private static void error(HttpExchange exchange, int status, String detail) throws IOException { ObjectNode body = JSON.createObjectNode().put("detail", detail); respond(exchange, status, body); }
    private static final class BadRequest extends RuntimeException { final int status; BadRequest(int status, String message) { super(message); this.status=status; } }
}
