package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.io.ByteArrayInputStream;
import java.io.RandomAccessFile;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import java.lang.reflect.Field;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;

class ControlServerTest {
  @Test void rejectsUnauthorizedRequestsAndReleasesBackupFromOwnerThread() throws Exception {
    try(var registry=new Registry(Files.createTempDirectory("control"),"key"); var engine=new StorageEngine(registry,Files.createTempDirectory("objects")); var control=new ControlServer(registry,engine,new InetSocketAddress("127.0.0.1",0),"token")) {
      control.start(); Field field=ControlServer.class.getDeclaredField("server");field.setAccessible(true);int port=((com.sun.net.httpserver.HttpServer)field.get(control)).getAddress().getPort();
      HttpClient client=HttpClient.newHttpClient();
      assertEquals(401,client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+port+"/v1/settings")).GET().build(),HttpResponse.BodyHandlers.ofString()).statusCode());
      var prepared=send(client,port,"POST","/v1/backup/prepare","{}"); assertEquals(200,prepared.statusCode()); assertTrue(registry.maintenanceLock().isWriteLocked());
      String lease=prepared.body().replaceAll(".*\\\"lease_id\\\":\\\"([^\\\"]+)\\\".*","$1");
      assertEquals(200,send(client,port,"POST","/v1/backup/release","{\"lease_id\":\""+lease+"\"}").statusCode());
      long deadline=System.nanoTime()+1_000_000_000L; while(registry.maintenanceLock().isWriteLocked() && System.nanoTime()<deadline) Thread.yield();
      assertFalse(registry.maintenanceLock().isWriteLocked());
    }
  }
  @Test void renameIsLockedAndPreservesSourceOnConflicts() throws Exception {
    try(var registry=new Registry(Files.createTempDirectory("control"),"key"); var engine=new StorageEngine(registry,Files.createTempDirectory("objects")); var control=new ControlServer(registry,engine,new InetSocketAddress("127.0.0.1",0),"token")) {
      control.start(); Field field=ControlServer.class.getDeclaredField("server");field.setAccessible(true);int port=((com.sun.net.httpserver.HttpServer)field.get(control)).getAddress().getPort(); HttpClient client=HttpClient.newHttpClient();
      assertEquals(200,send(client,port,"POST","/v1/workspaces/workspace/ensure","{}").statusCode());
      engine.workspaceStore("workspace").putBlob(PutObjectRequest.builder().bucket("default").key("source").contentType("text/plain").metadata(java.util.Map.of("tag","kept")).build(),new ByteArrayInputStream("source bytes".getBytes()));
      assertEquals(409,send(client,port,"POST","/v1/workspaces/workspace/buckets/default/rename","{\"key\":\"source\",\"new_key\":\"source\"}").statusCode());
      engine.workspaceStore("workspace").putBlob(PutObjectRequest.builder().bucket("default").key("destination").build(),new ByteArrayInputStream(new byte[]{1}));
      assertEquals(409,send(client,port,"POST","/v1/workspaces/workspace/buckets/default/rename","{\"key\":\"source\",\"new_key\":\"destination\"}").statusCode());
      try(var source=engine.workspaceStore("workspace").getBlob(GetObjectRequest.builder().bucket("default").key("source").build())) { assertEquals("source bytes",new String(source.readAllBytes())); }
      var renamed=send(client,port,"POST","/v1/workspaces/workspace/buckets/default/rename","{\"key\":\"source\",\"new_key\":\"renamed\"}"); assertEquals(200,renamed.statusCode()); assertTrue(renamed.body().contains("\"key\":\"renamed\""));
      assertFalse(engine.workspaceStore("workspace").blobExists("default","source")); assertTrue(engine.workspaceStore("workspace").blobExists("default","renamed"));
    }
  }
  @Test void importsOnlyManifestBoundStagingAndAcknowledgesExactCompletedGeneration() throws Exception {
    var root=Files.createTempDirectory("control");
    String generation="0123456789abcdef0123456789abcdef";
    var staged=Files.createDirectories(root.resolve("_legacy_imports/ws/").resolve(generation).resolve("buckets/uploads/docs"));
    Files.writeString(staged.resolve("report.txt"),"report");
    String fileHash=sha256("report".getBytes());
    String manifest="{\"version\":1,\"workspace_id\":\"ws\",\"generation\":\""+generation+"\",\"files\":[{\"path\":\"buckets/uploads/docs/report.txt\",\"size\":6,\"sha256\":\""+fileHash+"\"}]}";
    Files.writeString(staged.getParent().getParent().getParent().resolve("manifest.json"),manifest);
    String manifestHash=sha256(manifest.getBytes());
    try(var registry=new Registry(root,"key"); var engine=new StorageEngine(registry,root,"key"); var control=new ControlServer(registry,engine,new InetSocketAddress("127.0.0.1",0),"token")) {
      control.start(); Field field=ControlServer.class.getDeclaredField("server");field.setAccessible(true);int port=((com.sun.net.httpserver.HttpServer)field.get(control)).getAddress().getPort(); HttpClient client=HttpClient.newHttpClient();
      assertEquals(200,send(client,port,"POST","/v1/workspaces/ws/ensure","{\"buckets\":[{\"name\":\"uploads\"}]}" ).statusCode());
      var queued=send(client,port,"POST","/v1/workspaces/ws/legacy-import","{\"generation\":\""+generation+"\",\"manifest_sha256\":\""+manifestHash+"\"}");
      assertEquals(200,queued.statusCode(),queued.body());
      HttpResponse<String> status=null; long deadline=System.nanoTime()+5_000_000_000L;
      do { status=send(client,port,"GET","/v1/workspaces/ws/legacy-import",null); Thread.sleep(10); } while(!status.body().contains("\"state\":\"completed\"") && System.nanoTime()<deadline);
      assertEquals(200,status.statusCode()); assertTrue(status.body().contains("buckets/uploads/docs/report.txt"));
      assertEquals(200,send(client,port,"POST","/v1/workspaces/ws/legacy-import/gc","{\"generation\":\""+generation+"\",\"manifest_sha256\":\""+manifestHash+"\"}").statusCode());
      assertEquals(409,send(client,port,"POST","/v1/workspaces/ws/legacy-import/gc","{\"generation\":\""+generation+"\",\"manifest_sha256\":\"bad\"}").statusCode());
    }
  }
  @Test void concurrentDifferentStagedIdentitiesHaveExactlyOneWinner() throws Exception {
    var root=Files.createTempDirectory("control"); String first="0123456789abcdef0123456789abcdef", second="fedcba9876543210fedcba9876543210";
    try(var registry=new Registry(root,"key"); var engine=new StorageEngine(registry,root,"key"); var control=new ControlServer(registry,engine,new InetSocketAddress("127.0.0.1",0),"token")) {
      registry.mutate(state -> { var ws=state.withObject("workspaces").putObject("ws"); ws.put("workspace_id","ws").put("state","ready").put("legacy_import_state","pending"); });
      control.start(); Field field=ControlServer.class.getDeclaredField("server");field.setAccessible(true);int port=((com.sun.net.httpserver.HttpServer)field.get(control)).getAddress().getPort(); HttpClient client=HttpClient.newHttpClient();
      ExecutorService pool=Executors.newFixedThreadPool(2); CountDownLatch start=new CountDownLatch(1);
      Future<Integer> one=pool.submit(()->{start.await();return send(client,port,"POST","/v1/workspaces/ws/legacy-import","{\"generation\":\""+first+"\",\"manifest_sha256\":\""+"a".repeat(64)+"\"}").statusCode();});
      Future<Integer> two=pool.submit(()->{start.await();return send(client,port,"POST","/v1/workspaces/ws/legacy-import","{\"generation\":\""+second+"\",\"manifest_sha256\":\""+"b".repeat(64)+"\"}").statusCode();}); start.countDown();
      int a=one.get(),b=two.get(); pool.shutdownNow(); assertTrue((a==200&&b==409)||(a==409&&b==200));
    }
  }
  @Test void revokedWorkspaceIsNeverRestoredByResumedImport() throws Exception {
    var root=Files.createTempDirectory("control"); String generation="0123456789abcdef0123456789abcdef";
    try(var registry=new Registry(root,"key"); var engine=new StorageEngine(registry,root,"key")) {
      registry.mutate(state -> { var ws=state.withObject("workspaces").putObject("ws"); ws.put("workspace_id","ws").put("state","revoked").put("legacy_import_state","copying"); state.withObject("legacy_import_jobs").putObject("ws").put("workspace_id","ws").put("generation",generation).put("manifest_sha256","a".repeat(64)).put("state","copying"); });
       new LegacyImportService(registry,engine).run("ws"); assertEquals("revoked",registry.workspace("ws").path("state").asText());
       assertEquals("cancelled",LegacyImportService.job(registry.snapshot(),"ws").path("state").asText());
     }
   }
  @Test void adoptsPreUpgradeLegacyStatesOnlyWhenNoBackendMigrationIsActive() throws Exception {
    for (String[] state : new String[][]{{"ready", "pending"}, {"importing", "copying"}, {"importing", "failed"}}) {
      try(var registry=new Registry(Files.createTempDirectory("control"),"key"); var engine=new StorageEngine(registry,Files.createTempDirectory("objects")); var control=new ControlServer(registry,engine,new InetSocketAddress("127.0.0.1",0),"token")) {
        registry.mutate(snapshot -> { var ws=snapshot.withObject("workspaces").putObject("ws"); ws.put("workspace_id","ws").put("state",state[0]).put("legacy_import_state",state[1]); });
        control.start(); int port=port(control);
        assertEquals(200, send(HttpClient.newHttpClient(),port,"POST","/v1/workspaces/ws/legacy-import","{\"generation\":\"0123456789abcdef0123456789abcdef\",\"manifest_sha256\":\""+"a".repeat(64)+"\"}").statusCode());
        assertNotNull(LegacyImportService.job(registry.snapshot(), "ws"));
      }
    }
    try(var registry=new Registry(Files.createTempDirectory("control"),"key"); var engine=new StorageEngine(registry,Files.createTempDirectory("objects")); var control=new ControlServer(registry,engine,new InetSocketAddress("127.0.0.1",0),"token")) {
      registry.mutate(snapshot -> { var ws=snapshot.withObject("workspaces").putObject("ws"); ws.put("workspace_id","ws").put("state","ready").put("legacy_import_state","pending"); snapshot.withObject("migrations").putObject("backend").put("workspace_id","ws").put("state","copying"); });
      control.start();
      assertEquals(409, send(HttpClient.newHttpClient(),port(control),"POST","/v1/workspaces/ws/legacy-import","{\"generation\":\"0123456789abcdef0123456789abcdef\",\"manifest_sha256\":\""+"a".repeat(64)+"\"}").statusCode());
      assertNull(LegacyImportService.job(registry.snapshot(), "ws"));
    }
  }
  @Test void doesNotReplayCompletedOrRevokedLegacyImports() throws Exception {
    for (String state : new String[]{"completed", "revoked"}) {
      try(var registry=new Registry(Files.createTempDirectory("control"),"key"); var engine=new StorageEngine(registry,Files.createTempDirectory("objects")); var control=new ControlServer(registry,engine,new InetSocketAddress("127.0.0.1",0),"token")) {
        registry.mutate(snapshot -> { var ws=snapshot.withObject("workspaces").putObject("ws"); ws.put("workspace_id","ws").put("state",state.equals("revoked") ? "revoked" : "ready").put("legacy_import_state",state); });
        control.start();
        assertEquals(state.equals("revoked") ? 404 : 409, send(HttpClient.newHttpClient(),port(control),"POST","/v1/workspaces/ws/legacy-import","{\"generation\":\"0123456789abcdef0123456789abcdef\",\"manifest_sha256\":\""+"a".repeat(64)+"\"}").statusCode());
        assertNull(LegacyImportService.job(registry.snapshot(), "ws"));
      }
    }
  }
  @Test void rejectsOversizedVerificationEvidenceWithoutReadingIt() throws Exception {
    var root=Files.createTempDirectory("control"); String generation="0123456789abcdef0123456789abcdef";
    var evidence=root.resolve("_legacy_imports/ws").resolve(generation).resolve("verification.json"); Files.createDirectories(evidence.getParent());
    try (var file = new RandomAccessFile(evidence.toFile(), "rw")) { file.setLength((long) LegacyImporter.MAX_MANIFEST_OR_EVIDENCE_BYTES + 1); }
    try(var registry=new Registry(root,"key"); var engine=new StorageEngine(registry,root,"key"); var control=new ControlServer(registry,engine,new InetSocketAddress("127.0.0.1",0),"token")) {
      registry.mutate(snapshot -> { var ws=snapshot.withObject("workspaces").putObject("ws"); ws.put("workspace_id","ws").put("state","ready"); snapshot.withObject("legacy_import_jobs").putObject("ws").put("workspace_id","ws").put("generation",generation).put("manifest_sha256","a".repeat(64)).put("verification_sha256","b".repeat(64)).put("state","completed"); });
      control.start();
      var response=send(HttpClient.newHttpClient(),port(control),"GET","/v1/workspaces/ws/legacy-import",null);
      assertEquals(200,response.statusCode()); assertTrue(response.body().contains("\"verification_evidence_available\":false"));
    }
  }
  private static int port(ControlServer control) throws Exception { Field field=ControlServer.class.getDeclaredField("server");field.setAccessible(true);return ((com.sun.net.httpserver.HttpServer)field.get(control)).getAddress().getPort(); }
  private static String sha256(byte[] value)throws Exception{return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));}
  private static HttpResponse<String> send(HttpClient client,int port,String method,String path,String body)throws Exception {var builder=HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+port+path)).header("Authorization","Bearer token");if(body==null)builder.method(method,HttpRequest.BodyPublishers.noBody());else builder.method(method,HttpRequest.BodyPublishers.ofString(body));return client.send(builder.build(),HttpResponse.BodyHandlers.ofString());}
}
