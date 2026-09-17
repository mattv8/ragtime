package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;

/** Durable state transitions for staged legacy imports. */
final class LegacyImportService {
    private final Registry registry; private final LegacyImporter importer;
    LegacyImportService(Registry registry, StorageEngine engine) { this.registry = registry; this.importer = new LegacyImporter(registry, engine); }
    void run(String workspaceId) {
        ObjectNode job = job(workspaceId); if (job == null || !active(job)) return;
        try {
            transition(workspaceId, "copying", false, null);
            ObjectNode current = job(workspaceId); LegacyImporter.Result result = importer.importStaged(workspaceId, current.path("generation").asText(), current.path("manifest_sha256").asText());
            transition(workspaceId, "completed", true, result);
        } catch (Exception error) {
            registry.maintenanceLock().readLock().lock(); registry.workspaceLock(workspaceId).writeLock().lock();
            try { registry.mutate(state -> { ObjectNode current = job(state, workspaceId), workspace = workspace(state, workspaceId); if (current != null && workspace != null && !"revoked".equals(workspace.path("state").asText())) { current.put("state", "failed").put("error", "staged legacy import failed"); workspace.put("state", "importing").put("legacy_import_state", "failed"); } }); }
            finally { registry.workspaceLock(workspaceId).writeLock().unlock(); registry.maintenanceLock().readLock().unlock(); }
        }
    }
    private void transition(String workspaceId,String state,boolean complete,LegacyImporter.Result result) {
        registry.maintenanceLock().readLock().lock(); registry.workspaceLock(workspaceId).writeLock().lock();
        try { registry.mutate(snapshot -> { ObjectNode current=job(snapshot,workspaceId), workspace=workspace(snapshot,workspaceId); if(current==null || workspace==null || "revoked".equals(workspace.path("state").asText()) || !active(current) || !current.path("generation").asText().matches("[0-9a-f]{32}") || !current.path("manifest_sha256").asText().matches("[0-9a-f]{64}")) throw new IllegalStateException("legacy import is no longer eligible"); if(complete) { current.put("state","completed").put("verification_sha256",result.evidenceSha256()); workspace.put("state","ready").put("legacy_import_state","completed"); } else { current.put("state",state); workspace.put("state","importing").put("legacy_import_state",state); } }); }
        finally { registry.workspaceLock(workspaceId).writeLock().unlock(); registry.maintenanceLock().readLock().unlock(); }
    }
    static boolean active(JsonNode job) { String state=job.path("state").asText(); return state.equals("pending") || state.equals("copying"); }
    static ObjectNode job(ObjectNode state, String workspaceId) { JsonNode value=state.path("legacy_import_jobs").get(workspaceId); return value instanceof ObjectNode job ? job : null; }
    private ObjectNode job(String workspaceId) { return job(registry.snapshot(), workspaceId); }
    static ObjectNode workspace(ObjectNode state, String id) { JsonNode value=state.path("workspaces").get(id); return value instanceof ObjectNode workspace ? workspace : null; }
}
