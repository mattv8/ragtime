package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.gaul.s3proxy.AccessGrant;
import org.gaul.s3proxy.S3Proxy;
import org.gaul.s3proxy.auth.AuthenticationType;
import org.gaul.s3proxy.blobstore.BlobStore;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3Configuration;
import software.amazon.awssdk.services.s3.model.*;

class StorageEngineTest {
  @Test void indexedStoreUsesOpaqueBytesAndSeekPagination() throws Exception {
    Path root=Files.createTempDirectory("indexed"); IndexedLocalBlobStore store=new IndexedLocalBlobStore(root);
    for(int i=0;i<4;i++) store.putBlob(PutObjectRequest.builder().bucket("ragtime").key("p/"+i).contentType("text/plain").build(),new java.io.ByteArrayInputStream(("v"+i).getBytes()));
    ListObjectsV2Response first=store.list(ListObjectsV2Request.builder().bucket("ragtime").prefix("p/").maxKeys(2).build());
    assertEquals(List.of("p/0","p/1"),first.contents().stream().map(S3Object::key).toList()); assertTrue(first.isTruncated());
    ListObjectsV2Response second=store.list(ListObjectsV2Request.builder().bucket("ragtime").prefix("p/").maxKeys(2).continuationToken(first.nextContinuationToken()).build());
    assertEquals(List.of("p/2","p/3"),second.contents().stream().map(S3Object::key).toList());
    try(var paths=Files.list(root.resolve("bytes"))){assertTrue(paths.allMatch(p -> !p.getFileName().toString().contains("p")));}
  }
  @Test void delimiterPaginationAdvancesPastPrefixRange() throws Exception {
    Path root=Files.createTempDirectory("delim"); IndexedLocalBlobStore store=new IndexedLocalBlobStore(root);
    // Create folders where pagination breaks on a prefix boundary.
    // With delimiter="/", we list common prefixes only (no objects from inside the prefix).
    for(int i=1;i<=20;i++) {
      String folder = "dir-"+String.format("%02d", i);
      // Just put one object per folder to ensure it exists
      store.putBlob(PutObjectRequest.builder().bucket("ragtime").key(folder+"/file.txt").contentType("text/plain").build(),
        new java.io.ByteArrayInputStream(("data").getBytes()));
    }
    // First page: maxKeys=5 should give us 5 prefixes (dir-01/ through dir-05/).
    ListObjectsV2Response first=store.list(ListObjectsV2Request.builder().bucket("ragtime").delimiter("/").maxKeys(5).build());
    List<String> firstPrefixes = first.commonPrefixes().stream().map(CommonPrefix::prefix).toList();
    assertEquals(5, firstPrefixes.size(), "First page should have 5 prefixes");
    assertEquals(List.of("dir-01/","dir-02/","dir-03/","dir-04/","dir-05/"), firstPrefixes);
    assertTrue(first.isTruncated(), "Should be truncated since we have 20 prefixes");
    // Second page using the continuation token should start from dir-06/ and not repeat dir-05/*.
    ListObjectsV2Response second=store.list(ListObjectsV2Request.builder().bucket("ragtime").delimiter("/").maxKeys(5).continuationToken(first.nextContinuationToken()).build());
    List<String> secondPrefixes = second.commonPrefixes().stream().map(CommonPrefix::prefix).toList();
    assertEquals(5, secondPrefixes.size(), "Second page should have 5 prefixes");
    assertEquals(List.of("dir-06/","dir-07/","dir-08/","dir-09/","dir-10/"), secondPrefixes);
    assertFalse(secondPrefixes.stream().anyMatch(p -> firstPrefixes.contains(p)), "Second page should not repeat first page prefixes");
  }
  @Test void signedGatewayIsTenantScopedAndRejectsUnsigned() throws Exception {
    Path root=Files.createTempDirectory("gateway"); Registry registry=new Registry(root,"test-key");
    registry.mutate(s -> { workspace(s,"one","ak-one","secret-one","alpha","bid-a"); workspace(s,"two","ak-two","secret-two","beta","bid-b"); });
    StorageEngine engine=new StorageEngine(registry,root,"test-key"); BlobStore base=engine.physicalStore("local");
    S3Proxy proxy=S3Proxy.builder().endpoint(URI.create("http://127.0.0.1:0")).blobStore(base).awsAuthentication(AuthenticationType.AWS_V4,"unused","unused").build();
    proxy.setBlobStoreLocator((id,bucket,key) -> { ObjectNode w=engine.registry().findIdentity(id); if(w==null)return null; if(bucket!=null&&!bucket.isEmpty()&&w.path("buckets").findValuesAsText("name").stream().noneMatch(bucket::equals))return null; return new AccessGrant(w.path("secret_access_key").asText(),engine.workspaceStore(w.path("workspace_id").asText())); });
    proxy.start(); try {
      S3Client one=client(proxy.getPort(),"ak-one","secret-one"), two=client(proxy.getPort(),"ak-two","secret-two"), bad=client(proxy.getPort(),"wrong","wrong");
      one.putObject(PutObjectRequest.builder().bucket("alpha").key("p/a").build(),RequestBody.fromString("hello"));
      assertEquals("hello",one.getObjectAsBytes(GetObjectRequest.builder().bucket("alpha").key("p/a").build()).asUtf8String());
      assertThrows(S3Exception.class,()->two.getObjectAsBytes(GetObjectRequest.builder().bucket("alpha").key("p/a").build()));
      assertThrows(S3Exception.class,()->bad.listBuckets());
      assertEquals(403,java.net.http.HttpClient.newHttpClient().send(java.net.http.HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+proxy.getPort()+"/")).GET().build(),java.net.http.HttpResponse.BodyHandlers.ofString()).statusCode());
    } finally { proxy.stop(); engine.close(); registry.close(); }
  }
  private static void workspace(ObjectNode s,String id,String access,String secret,String name,String bucketId){ObjectNode w=s.withObject("workspaces").putObject(id);w.put("workspace_id",id);w.put("access_key_id",access);w.put("secret_access_key",secret);w.put("state","ready");w.put("default_bucket_name",name);ObjectNode b=w.putArray("buckets").addObject();b.put("id",bucketId);b.put("name",name);b.put("backend_id","local");}
  private static S3Client client(int port,String key,String secret){return S3Client.builder().endpointOverride(URI.create("http://127.0.0.1:"+port)).region(Region.US_EAST_1).serviceConfiguration(S3Configuration.builder().pathStyleAccessEnabled(true).build()).credentialsProvider(StaticCredentialsProvider.create(AwsBasicCredentials.create(key,secret))).build();}
}
