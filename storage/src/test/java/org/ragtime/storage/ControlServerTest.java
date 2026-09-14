package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.io.ByteArrayInputStream;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import java.lang.reflect.Field;
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
  private static HttpResponse<String> send(HttpClient client,int port,String method,String path,String body)throws Exception {var builder=HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+port+path)).header("Authorization","Bearer token");if(body==null)builder.method(method,HttpRequest.BodyPublishers.noBody());else builder.method(method,HttpRequest.BodyPublishers.ofString(body));return client.send(builder.build(),HttpResponse.BodyHandlers.ofString());}
}
