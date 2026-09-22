package org.ragtime.storage;

import com.fasterxml.jackson.databind.node.ObjectNode;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.gaul.s3proxy.AccessGrant;
import org.gaul.s3proxy.S3Proxy;
import org.gaul.s3proxy.auth.AuthenticationType;
import org.gaul.s3proxy.blobstore.BlobStore;

/** Storage process entrypoint.  The managed key is bootstrap material, never generated here. */
public final class Main {
  private Main() { }
  public static void main(String[] args) throws Exception {
    Path root=Path.of(env("STORAGE_ROOT","/data/_userspace/_object_storage"));
    Path keyFile=Path.of(env("STORAGE_KEY_FILE","/run/ragtime-keystore/.encryption_key"));
    String key=waitForEncryptionKey(keyFile);
    Registry registry=new Registry(root,key); StorageEngine engine=new StorageEngine(registry,root,key);
    BlobStore denyAll=new DenyAllBlobStore(engine.physicalStore("local"));
    S3Proxy proxy=S3Proxy.builder().endpoint(URI.create("http://0.0.0.0:"+env("S3_PORT","9000")))
        .blobStore(denyAll).awsAuthentication(AuthenticationType.AWS_V4,"ragtime-deny-all","ragtime-deny-all")
        .ignoreUnknownHeaders(false).build();
    proxy.setBlobStoreLocator((identity,container,blob)->locate(engine,identity,container));
    ControlServer control=new ControlServer(registry,engine,new InetSocketAddress("0.0.0.0",Integer.parseInt(env("STORAGE_CONTROL_PORT","9001"))),controlToken(key));
    Runtime.getRuntime().addShutdownHook(new Thread(() -> close(proxy,control,engine,registry)));
    control.start(); proxy.start();
  }
  private static String waitForEncryptionKey(Path keyFile) throws Exception {
    long deadline = System.currentTimeMillis() + 30_000; // 30 second timeout
    while (System.currentTimeMillis() < deadline) {
      try {
        String key = Files.readString(keyFile, StandardCharsets.UTF_8).strip();
        if (!key.isEmpty()) return key;
      } catch (Exception ignored) {
        // File not found or unreadable; will retry
      }
      Thread.sleep(500); // Retry every 500ms
    }
    throw new IllegalStateException("encryption key not found at " + keyFile + " within 30s");
  }
  private static AccessGrant locate(StorageEngine engine,String identity,String container) {
    if(identity==null) return null;
    ObjectNode w=engine.registry().findIdentity(identity); if(w==null) return null;
    // Allow access for valid identities regardless of state; Namespace/control layer enforces mutation fences.
    if(container!=null && !container.isEmpty()) { boolean found=false; for(var b:w.path("buckets")) if(container.equals(b.path("name").asText())) { found=true; break; } if(!found) return null; }
    String secret=w.path("secret_access_key").asText(); return secret.isBlank()?null:new AccessGrant(secret,engine.workspaceStore(w.path("workspace_id").asText()));
  }
  private static String controlToken(String key) throws Exception { Mac mac=Mac.getInstance("HmacSHA256"); mac.init(new SecretKeySpec(key.getBytes(StandardCharsets.UTF_8),"HmacSHA256")); return java.util.HexFormat.of().formatHex(mac.doFinal("ragtime-object-storage-control-v1".getBytes(StandardCharsets.UTF_8))); }
  private static String env(String name,String fallback) { String value=System.getenv(name); return value==null||value.isBlank()?fallback:value; }
  private static void close(S3Proxy proxy,ControlServer control,StorageEngine engine,Registry registry) { try { proxy.stop(); } catch(Exception ignored) {} try { control.close(); } catch(Exception ignored) {} engine.close(); registry.close(); }
}
