package org.ragtime.storage;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.sql.*;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.gaul.s3proxy.blobstore.BlobStore;
import org.gaul.s3proxy.blobstore.SdkResponses;
import org.gaul.s3proxy.blobstore.domain.MultipartUpload;
import software.amazon.awssdk.awscore.exception.AwsErrorDetails;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.services.s3.model.*;

/** Local object storage backed by an ordered SQLite index and opaque payload filenames. */
final class IndexedLocalBlobStore implements BlobStore {
  private static final long BUSY_TIMEOUT_MS = 30_000;
  private static final long MIN_PART_SIZE = 5L * 1024 * 1024;
  private static final ObjectMapper JSON = new ObjectMapper();
  private final Path bytes;
  private final Path staging;
  private final String jdbc;
  private final byte[] tokenKey;
  private final ReentrantLock[] locks = new ReentrantLock[128];

  IndexedLocalBlobStore(Path root) {
    try {
      bytes = Files.createDirectories(root.resolve("bytes"));
      staging = Files.createDirectories(root.resolve("staging"));
      jdbc = "jdbc:sqlite:" + root.resolve("objects.sqlite");
      for (int i = 0; i < locks.length; i++) locks[i] = new ReentrantLock();
      try (Connection c = connection(); Statement s = c.createStatement()) {
        s.execute("PRAGMA journal_mode=WAL");
        s.execute("CREATE TABLE IF NOT EXISTS objects(bucket TEXT NOT NULL,key TEXT NOT NULL,file TEXT NOT NULL,size INTEGER NOT NULL,etag TEXT NOT NULL,sha256 TEXT NOT NULL,type TEXT,metadata TEXT,cache_control TEXT,content_encoding TEXT,content_disposition TEXT,content_language TEXT,expires INTEGER,modified INTEGER NOT NULL,PRIMARY KEY(bucket,key))");
        s.execute("CREATE TABLE IF NOT EXISTS uploads(id TEXT PRIMARY KEY,bucket TEXT NOT NULL,key TEXT NOT NULL,type TEXT,metadata TEXT,created INTEGER NOT NULL)");
        s.execute("CREATE TABLE IF NOT EXISTS parts(upload_id TEXT NOT NULL,part INTEGER NOT NULL,file TEXT NOT NULL,size INTEGER NOT NULL,etag TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(upload_id,part))");
        s.execute("CREATE TABLE IF NOT EXISTS store_meta(name TEXT PRIMARY KEY,value BLOB NOT NULL)");
        addColumn(s, "objects", "cache_control TEXT"); addColumn(s, "objects", "content_encoding TEXT");
        addColumn(s, "objects", "content_disposition TEXT"); addColumn(s, "objects", "content_language TEXT"); addColumn(s, "objects", "expires INTEGER");
        addColumn(s, "objects", "metadata TEXT"); addColumn(s, "uploads", "metadata TEXT");
        addColumn(s, "uploads", "created INTEGER NOT NULL DEFAULT 0");
        addColumn(s, "parts", "sha256 TEXT");
      }
      tokenKey = tokenKey();
    } catch (Exception e) { throw new IllegalStateException("cannot initialize indexed object store", e); }
  }

  private static void addColumn(Statement s, String table, String definition) { try { s.execute("ALTER TABLE " + table + " ADD COLUMN " + definition); } catch (SQLException ignored) {} }
  private Connection connection() throws SQLException {
    java.util.Properties properties = new java.util.Properties();
    properties.setProperty("transaction_mode", "IMMEDIATE");
    properties.setProperty("busy_timeout", Long.toString(BUSY_TIMEOUT_MS));
    return DriverManager.getConnection(jdbc, properties);
  }
  private byte[] tokenKey() throws Exception {
    try (Connection c = connection(); PreparedStatement q = c.prepareStatement("SELECT value FROM store_meta WHERE name='token_key'")) {
      try (ResultSet r = q.executeQuery()) { if (r.next()) return r.getBytes(1); }
      byte[] value = new byte[32]; new SecureRandom().nextBytes(value);
      try (PreparedStatement i = c.prepareStatement("INSERT INTO store_meta(name,value) VALUES('token_key',?)")) { i.setBytes(1, value); i.executeUpdate(); }
      return value;
    }
  }
  private static String random() { return UUID.randomUUID().toString().replace("-", ""); }
  private static String hex(byte[] value) { return HexFormat.of().formatHex(value); }
  private static S3Exception error(String code, int status, String message) { return (S3Exception) S3Exception.builder().statusCode(status).message(message).awsErrorDetails(AwsErrorDetails.builder().errorCode(code).errorMessage(message).build()).build(); }
  private static S3Exception noSuchKey() { return error("NoSuchKey", 404, "The specified key does not exist."); }
  private static S3Exception noSuchUpload() { return error("NoSuchUpload", 404, "The specified upload does not exist."); }
  private ReentrantLock lock(String bucket, String key) { return locks[Math.floorMod(Objects.hash(bucket, key), locks.length)]; }
  private Path payload(String name, Path directory) { if (!name.matches("[0-9a-f]{32}")) throw new IllegalStateException("invalid payload id"); return directory.resolve(name); }

  private record Obj(String file, long size, String etag, String sha, String type, Map<String,String> metadata, String cacheControl, String encoding, String disposition, String language, Long expires, long modified) {}
  private record Opened(Obj object, InputStream input) implements AutoCloseable { public void close() throws IOException { input.close(); } }
  private Opened open(String bucket, String key) throws IOException {
    ReentrantLock guard = lock(bucket, key);
    guard.lock();
    try { Obj object = object(bucket, key); return new Opened(object, Files.newInputStream(payload(object.file, bytes))); }
    finally { guard.unlock(); }
  }
  private Obj object(Connection c, String bucket, String key) throws SQLException {
    try (PreparedStatement s = c.prepareStatement("SELECT file,size,etag,sha256,type,metadata,cache_control,content_encoding,content_disposition,content_language,expires,modified FROM objects WHERE bucket=? AND key=?")) {
      s.setString(1, bucket); s.setString(2, key); try (ResultSet r = s.executeQuery()) {
        if (!r.next()) return null;
        return new Obj(r.getString(1), r.getLong(2), r.getString(3), r.getString(4), r.getString(5), metadata(r.getString(6)), r.getString(7), r.getString(8), r.getString(9), r.getString(10), r.getObject(11) == null ? null : r.getLong(11), r.getLong(12));
      }
    }
  }
  private Obj object(String bucket, String key) { try (Connection c = connection()) { Obj o = object(c, bucket, key); if (o == null) throw noSuchKey(); return o; } catch (SQLException e) { throw new IllegalStateException(e); } }
  private static Map<String,String> metadata(String raw) { try { return raw == null ? Map.of() : Map.copyOf(JSON.readValue(raw, new TypeReference<Map<String,String>>() {})); } catch (Exception e) { throw new IllegalStateException("invalid stored metadata", e); } }
  private static String metadata(Map<String,String> value) { try { return value == null || value.isEmpty() ? null : JSON.writeValueAsString(value); } catch (Exception e) { throw new IllegalArgumentException("metadata is not serializable", e); } }
  private static String quoted(String etag) { return "\"" + etag + "\""; }
  private static boolean etagMatches(String supplied, String etag) { if (supplied == null) return false; for (String candidate : supplied.split(",")) if (candidate.trim().equals("*") || candidate.trim().replace("\"", "").equals(etag)) return true; return false; }

  private record Staged(Path path, long size, String md5, String sha256) {}
  private Staged stage(InputStream input) {
    Path path = staging.resolve(random());
    try (InputStream in = input; OutputStream out = Files.newOutputStream(path)) {
      MessageDigest md5 = MessageDigest.getInstance("MD5"), sha = MessageDigest.getInstance("SHA-256");
      byte[] buffer = new byte[8192]; long size = 0; int n;
      while ((n = in.read(buffer)) >= 0) { out.write(buffer, 0, n); md5.update(buffer, 0, n); sha.update(buffer, 0, n); size += n; }
      return new Staged(path, size, hex(md5.digest()), hex(sha.digest()));
    } catch (Exception e) { delete(path); throw new IllegalStateException("cannot stage object", e); }
  }
  private static void delete(Path path) { if (path != null) try { Files.deleteIfExists(path); } catch (IOException ignored) {} }

  private PutObjectResponse publish(PutObjectRequest request, Staged staged) {
    ReentrantLock lock = lock(request.bucket(), request.key()); lock.lock(); String newFile = random(); Path finalPath = payload(newFile, bytes); Obj old = null;
    try {
      Files.move(staged.path, finalPath, StandardCopyOption.ATOMIC_MOVE);
      try (Connection c = connection()) {
        c.setAutoCommit(false); old = object(c, request.bucket(), request.key()); checkWriteConditions(request.ifMatch(), request.ifNoneMatch(), old);
        try (PreparedStatement s = c.prepareStatement("INSERT INTO objects(bucket,key,file,size,etag,sha256,type,metadata,cache_control,content_encoding,content_disposition,content_language,expires,modified) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(bucket,key) DO UPDATE SET file=excluded.file,size=excluded.size,etag=excluded.etag,sha256=excluded.sha256,type=excluded.type,metadata=excluded.metadata,cache_control=excluded.cache_control,content_encoding=excluded.content_encoding,content_disposition=excluded.content_disposition,content_language=excluded.content_language,expires=excluded.expires,modified=excluded.modified")) {
          s.setString(1, request.bucket()); s.setString(2, request.key()); s.setString(3, newFile); s.setLong(4, staged.size); s.setString(5, staged.md5); s.setString(6, staged.sha256); s.setString(7, request.contentType()); s.setString(8, metadata(request.metadata())); s.setString(9, request.cacheControl()); s.setString(10, request.contentEncoding()); s.setString(11, request.contentDisposition()); s.setString(12, request.contentLanguage()); if (request.expires() == null) s.setNull(13, Types.BIGINT); else s.setLong(13, request.expires().toEpochMilli()); s.setLong(14, System.currentTimeMillis()); s.executeUpdate();
        }
        c.commit();
      }
      if (old != null) delete(payload(old.file, bytes));
      return SdkResponses.putResponse(staged.md5);
    } catch (Exception e) { delete(finalPath); if (e instanceof S3Exception s3) throw s3; throw new IllegalStateException("object publication failed", e); }
    finally { delete(staged.path); lock.unlock(); }
  }
  private static void checkWriteConditions(String ifMatch, String ifNoneMatch, Obj current) {
    if (ifMatch != null && (current == null || !etagMatches(ifMatch, current.etag))) throw error("PreconditionFailed", 412, "ETag does not match");
    if (ifNoneMatch != null && ((ifNoneMatch.equals("*") && current != null) || (current != null && etagMatches(ifNoneMatch, current.etag)))) throw error("PreconditionFailed", 412, "Object already exists");
  }

  @Override public ListBucketsResponse list() { return ListBucketsResponse.builder().buckets(List.of(Bucket.builder().name("ragtime").build())).build(); }
  @Override public HeadBucketResponse headBucket(HeadBucketRequest r) { return HeadBucketResponse.builder().build(); }
  @Override public CreateBucketResponse createContainer(CreateBucketRequest r) { return CreateBucketResponse.builder().build(); }
  @Override public BucketCannedACL getContainerAccess(String c) { return BucketCannedACL.PRIVATE; }
  @Override public void setContainerAccess(String c, BucketCannedACL a) {}
  @Override public void deleteBucket(String c) { throw error("AccessDenied", 403, "bucket deletion is control-plane only"); }
  @Override public boolean blobExists(String b, String k) { try { object(b, k); return true; } catch (S3Exception e) { return false; } }
  @Override public PutObjectResponse putBlob(PutObjectRequest r, InputStream in) { return publish(r, stage(in)); }
  private static HeadObjectResponse.Builder head(Obj o) { HeadObjectResponse.Builder b = HeadObjectResponse.builder().contentLength(o.size).eTag(quoted(o.etag)).contentType(o.type).lastModified(Instant.ofEpochMilli(o.modified)).metadata(o.metadata).cacheControl(o.cacheControl).contentEncoding(o.encoding).contentDisposition(o.disposition).contentLanguage(o.language); if (o.expires != null) b.expires(Instant.ofEpochMilli(o.expires)); return b; }
  @Override public HeadObjectResponse blobMetadata(HeadObjectRequest r) { Obj o = object(r.bucket(), r.key()); checkConditions(r.ifMatch(), r.ifNoneMatch(), r.ifModifiedSince(), r.ifUnmodifiedSince(), o); return head(o).build(); }
  @Override public ResponseInputStream<GetObjectResponse> getBlob(GetObjectRequest r) {
    Opened opened;
    try { opened = open(r.bucket(), r.key()); } catch (IOException error) { throw new UncheckedIOException(error); }
    try {
      Obj o = opened.object;
      checkConditions(r.ifMatch(), r.ifNoneMatch(), r.ifModifiedSince(), r.ifUnmodifiedSince(), o);
      Range range = Range.parse(r.range(), o.size);
      InputStream input = new LimitedInputStream(opened.input, range.start, range.length);
      GetObjectResponse.Builder b = GetObjectResponse.builder().contentLength(range.length).eTag(quoted(o.etag)).contentType(o.type)
          .lastModified(Instant.ofEpochMilli(o.modified)).metadata(o.metadata).cacheControl(o.cacheControl)
          .contentEncoding(o.encoding).contentDisposition(o.disposition).contentLanguage(o.language);
      if (range.ranged) b.contentRange("bytes " + range.start + "-" + range.end + "/" + o.size);
      if (o.expires != null) b.expires(Instant.ofEpochMilli(o.expires));
      return SdkResponses.getResponse(b.build(), input);
    } catch (Exception error) {
      try { opened.close(); } catch (IOException ignored) { }
      if (error instanceof RuntimeException runtime) throw runtime;
      throw new IllegalStateException(error);
    }
  }
  private static void checkConditions(String ifMatch, String ifNone, Instant modifiedSince, Instant unmodifiedSince, Obj o) { if (ifMatch != null && !etagMatches(ifMatch, o.etag)) throw error("PreconditionFailed", 412, "ETag does not match"); if (ifNone != null && etagMatches(ifNone, o.etag)) throw error("NotModified", 304, "Not Modified"); if (modifiedSince != null && o.modified <= modifiedSince.toEpochMilli()) throw error("NotModified", 304, "Not Modified"); if (unmodifiedSince != null && o.modified > unmodifiedSince.toEpochMilli()) throw error("PreconditionFailed", 412, "Precondition Failed"); }
  @Override public DeleteObjectResponse removeBlob(DeleteObjectRequest r) {
    ReentrantLock lock = lock(r.bucket(), r.key()); lock.lock(); Obj old = null;
    try (Connection c = connection()) { c.setAutoCommit(false); old = object(c, r.bucket(), r.key()); if (old != null) { checkWriteConditions(r.ifMatch(), null, old); try (PreparedStatement d = c.prepareStatement("DELETE FROM objects WHERE bucket=? AND key=?")) { d.setString(1, r.bucket()); d.setString(2, r.key()); d.executeUpdate(); } } c.commit(); } catch (SQLException e) { throw new IllegalStateException(e); } finally { lock.unlock(); }
    if (old != null) delete(payload(old.file, bytes)); return DeleteObjectResponse.builder().build();
  }
  @Override public CopyObjectResponse copyBlob(CopyObjectRequest r) {
    try (Opened opened = open(r.sourceBucket(), r.sourceKey())) {
      Obj source = opened.object;
      checkConditions(r.copySourceIfMatch(), r.copySourceIfNoneMatch(), r.copySourceIfModifiedSince(), r.copySourceIfUnmodifiedSince(), source);
      PutObjectRequest.Builder b = PutObjectRequest.builder().bucket(r.destinationBucket()).key(r.destinationKey()).contentType(source.type)
          .metadata(source.metadata).cacheControl(source.cacheControl).contentEncoding(source.encoding).contentDisposition(source.disposition)
          .contentLanguage(source.language).ifMatch(r.ifMatch()).ifNoneMatch(r.ifNoneMatch());
      if (r.metadataDirective() == MetadataDirective.REPLACE) b.metadata(r.metadata()).contentType(r.contentType())
          .cacheControl(r.cacheControl()).contentEncoding(r.contentEncoding()).contentDisposition(r.contentDisposition()).contentLanguage(r.contentLanguage());
      if (source.expires != null) b.expires(Instant.ofEpochMilli(source.expires));
      PutObjectResponse put = publish(b.build(), stage(opened.input));
      return SdkResponses.copyResponse(put.eTag(), Instant.now());
    } catch (IOException e) { throw new UncheckedIOException(e); }
  }

  private String token(String cursor) { try { byte[] data = cursor.getBytes(java.nio.charset.StandardCharsets.UTF_8); Mac mac = Mac.getInstance("HmacSHA256"); mac.init(new SecretKeySpec(tokenKey, "HmacSHA256")); byte[] sig = mac.doFinal(data); return Base64.getUrlEncoder().withoutPadding().encodeToString(data) + "." + Base64.getUrlEncoder().withoutPadding().encodeToString(sig); } catch (GeneralSecurityException e) { throw new IllegalStateException(e); } }
  private String cursor(String token) { try { String[] pieces = token.split("\\.", -1); if (pieces.length != 2) throw new IllegalArgumentException(); byte[] data = Base64.getUrlDecoder().decode(pieces[0]), signature = Base64.getUrlDecoder().decode(pieces[1]); Mac mac = Mac.getInstance("HmacSHA256"); mac.init(new SecretKeySpec(tokenKey, "HmacSHA256")); if (!MessageDigest.isEqual(signature, mac.doFinal(data))) throw new IllegalArgumentException(); return new String(data, java.nio.charset.StandardCharsets.UTF_8); } catch (Exception e) { throw error("InvalidArgument", 400, "Invalid continuation token"); } }
  private static String successor(String value) {
    for (int end = value.length(); end > 0;) {
      int codePoint = value.codePointBefore(end);
      int start = end - Character.charCount(codePoint);
      if (codePoint < Character.MAX_CODE_POINT) return value.substring(0, start) + new String(Character.toChars(codePoint + 1));
      end = start;
    }
    return null;
  }
  @Override public ListObjectsV2Response list(ListObjectsV2Request r) {
    String prefix = Optional.ofNullable(r.prefix()).orElse(""), delimiter = Optional.ofNullable(r.delimiter()).orElse(""); int max = r.maxKeys() == null ? 1000 : r.maxKeys(); if (max < 0) throw error("InvalidArgument", 400, "max-keys must not be negative"); if (max == 0) return ListObjectsV2Response.builder().contents(List.of()).commonPrefixes(List.of()).isTruncated(false).build();
    String lower = r.continuationToken() != null ? cursor(r.continuationToken()) : r.startAfter() != null ? r.startAfter() + "\0" : prefix;
    if (Arrays.compareUnsigned(lower.getBytes(java.nio.charset.StandardCharsets.UTF_8), prefix.getBytes(java.nio.charset.StandardCharsets.UTF_8)) < 0) lower = prefix;
    String upper = successor(prefix); List<S3Object> objects = new ArrayList<>(); List<CommonPrefix> prefixes = new ArrayList<>(); String next = null; boolean more = false;
    try (Connection c = connection()) {
      while (objects.size() + prefixes.size() < max) {
        String sql = upper == null ? "SELECT key,size,etag,modified FROM objects WHERE bucket=? AND key>=? ORDER BY key LIMIT 1" : "SELECT key,size,etag,modified FROM objects WHERE bucket=? AND key>=? AND key<? ORDER BY key LIMIT 1";
         try (PreparedStatement s = c.prepareStatement(sql)) { s.setString(1, r.bucket()); s.setString(2, lower); if (upper != null) s.setString(3, upper); try (ResultSet row = s.executeQuery()) { if (!row.next()) break; String key = row.getString(1); int position = delimiter.isEmpty() ? -1 : key.indexOf(delimiter, prefix.length()); if (position >= 0) { String common = key.substring(0, position + delimiter.length()); prefixes.add(CommonPrefix.builder().prefix(common).build()); String after = successor(common); if (after == null) break; lower = after; next = after; } else { objects.add(S3Object.builder().key(key).size(row.getLong(2)).eTag(quoted(row.getString(3))).lastModified(Instant.ofEpochMilli(row.getLong(4))).build()); lower = key + "\0"; next = lower; } } }
      }
      if (objects.size() + prefixes.size() == max) { String sql = upper == null ? "SELECT 1 FROM objects WHERE bucket=? AND key>=? LIMIT 1" : "SELECT 1 FROM objects WHERE bucket=? AND key>=? AND key<? LIMIT 1"; try (PreparedStatement s = c.prepareStatement(sql)) { s.setString(1, r.bucket()); s.setString(2, lower); if (upper != null) s.setString(3, upper); try (ResultSet x = s.executeQuery()) { more = x.next(); } } }
    } catch (SQLException e) { throw new IllegalStateException(e); }
    return ListObjectsV2Response.builder().contents(objects).commonPrefixes(prefixes).isTruncated(more).nextContinuationToken(more ? token(next == null ? lower : next) : null).build();
  }

  private record Range(long start, long end, long length, boolean ranged) { static Range parse(String value, long size) { if (value == null) return new Range(0, Math.max(0, size - 1), size, false); try { if (!value.startsWith("bytes=") || value.indexOf(',') >= 0) throw new IllegalArgumentException(); String[] p = value.substring(6).split("-", -1); if (p.length != 2) throw new IllegalArgumentException(); long start, end; if (p[0].isEmpty()) { long suffix = Long.parseLong(p[1]); if (suffix <= 0) throw new IllegalArgumentException(); start = Math.max(0, size - suffix); end = size - 1; } else { start = Long.parseLong(p[0]); end = p[1].isEmpty() ? size - 1 : Math.min(Long.parseLong(p[1]), size - 1); } if (size == 0 || start < 0 || start >= size || end < start) throw new IllegalArgumentException(); return new Range(start, end, end - start + 1, true); } catch (RuntimeException e) { throw error("InvalidRange", 416, "The requested range is not satisfiable"); } } }
  private static final class LimitedInputStream extends InputStream { private final InputStream source; private long remaining; LimitedInputStream(InputStream source, long skip, long remaining) throws IOException { this.source = source; this.remaining = remaining; while (skip > 0) { long n = source.skip(skip); if (n <= 0) { if (source.read() < 0) throw new EOFException(); n = 1; } skip -= n; } } @Override public int read() throws IOException { if (remaining == 0) return -1; int n = source.read(); if (n >= 0) remaining--; return n; } @Override public int read(byte[] b, int off, int len) throws IOException { if (remaining == 0) return -1; int n = source.read(b, off, (int) Math.min(len, remaining)); if (n > 0) remaining -= n; return n; } @Override public void close() throws IOException { source.close(); } }

  @Override public ObjectCannedACL getBlobAccess(String c, String n) { return ObjectCannedACL.PRIVATE; }
  @Override public void setBlobAccess(String c, String n, ObjectCannedACL a) {}
  private record Upload(String id, String bucket, String key, String type, Map<String,String> metadata) {}
  private Upload upload(Connection c, MultipartUpload m) throws SQLException { try (PreparedStatement s = c.prepareStatement("SELECT bucket,key,type,metadata FROM uploads WHERE id=?")) { s.setString(1, m.id()); try (ResultSet r = s.executeQuery()) { if (!r.next()) throw noSuchUpload(); if (!r.getString(1).equals(m.containerName()) || !r.getString(2).equals(m.blobName())) throw noSuchUpload(); return new Upload(m.id(), r.getString(1), r.getString(2), r.getString(3), metadata(r.getString(4))); } } }
  @Override public MultipartUpload initiateMultipartUpload(CreateMultipartUploadRequest r) { String id = random(); try (Connection c = connection(); PreparedStatement s = c.prepareStatement("INSERT INTO uploads(id,bucket,key,type,metadata,created) VALUES(?,?,?,?,?,?)")) { s.setString(1,id);s.setString(2,r.bucket());s.setString(3,r.key());s.setString(4,r.contentType());s.setString(5,metadata(r.metadata()));s.setLong(6,System.currentTimeMillis());s.executeUpdate(); return new MultipartUpload(id,r); } catch(SQLException e) { throw new IllegalStateException(e); } }
  @Override public UploadPartResponse uploadMultipartPart(MultipartUpload m, UploadPartRequest r, InputStream in) { if (r.partNumber() == null || r.partNumber() < 1 || r.partNumber() > 10_000) throw error("InvalidPart",400,"part number must be between 1 and 10000"); Staged staged = stage(in); ReentrantLock lock = lock(m.containerName(), m.blobName()); lock.lock(); String oldFile = null; try (Connection c=connection()) { c.setAutoCommit(false); upload(c,m); try(PreparedStatement q=c.prepareStatement("SELECT file FROM parts WHERE upload_id=? AND part=?")){q.setString(1,m.id());q.setInt(2,r.partNumber());try(ResultSet x=q.executeQuery()){if(x.next())oldFile=x.getString(1);}} try(PreparedStatement s=c.prepareStatement("INSERT INTO parts(upload_id,part,file,size,etag,sha256) VALUES(?,?,?,?,?,?) ON CONFLICT(upload_id,part) DO UPDATE SET file=excluded.file,size=excluded.size,etag=excluded.etag,sha256=excluded.sha256")){s.setString(1,m.id());s.setInt(2,r.partNumber());s.setString(3,staged.path.getFileName().toString());s.setLong(4,staged.size);s.setString(5,staged.md5);s.setString(6,staged.sha256);s.executeUpdate();} c.commit(); if(oldFile!=null)delete(payload(oldFile,staging)); return SdkResponses.uploadedPart(staged.md5); } catch(Exception e){delete(staged.path);if(e instanceof S3Exception s3)throw s3;throw new IllegalStateException(e);} finally {lock.unlock();} }
  @Override public List<Part> listMultipartUpload(MultipartUpload m) { List<Part> out=new ArrayList<>();try(Connection c=connection()){upload(c,m);try(PreparedStatement s=c.prepareStatement("SELECT part,size,etag FROM parts WHERE upload_id=? ORDER BY part")){s.setString(1,m.id());try(ResultSet r=s.executeQuery()){while(r.next())out.add(SdkResponses.part(r.getInt(1),r.getLong(2),r.getString(3),null));}}}catch(SQLException e){throw new IllegalStateException(e);}return out; }
  @Override public CompleteMultipartUploadResponse completeMultipartUpload(MultipartUpload m, CompleteMultipartUploadRequest r) {
    ReentrantLock lock = lock(m.containerName(), m.blobName()); lock.lock(); Path joined = staging.resolve(random());
    try (Connection c = connection()) {
      c.setAutoCommit(false); Upload upload = upload(c, m);
      List<CompletedPart> manifest = r.multipartUpload() == null ? List.of() : r.multipartUpload().parts();
      if (manifest == null || manifest.isEmpty()) throw error("InvalidRequest", 400, "A multipart manifest is required");
      int previous = 0; List<String> files = new ArrayList<>(); MessageDigest multipart = MessageDigest.getInstance("MD5");
      try (OutputStream out = Files.newOutputStream(joined); PreparedStatement q = c.prepareStatement("SELECT file,size,etag FROM parts WHERE upload_id=? AND part=?")) {
        for (int index = 0; index < manifest.size(); index++) {
          CompletedPart wanted = manifest.get(index);
          if (wanted.partNumber() == null || wanted.partNumber() <= previous) throw error("InvalidPartOrder", 400, "Parts must be ordered by part number");
          previous = wanted.partNumber(); q.setString(1, m.id()); q.setInt(2, previous);
          try (ResultSet row = q.executeQuery()) {
            if (!row.next() || wanted.eTag() == null || !etagMatches(wanted.eTag(), row.getString(3))) throw error("InvalidPart", 400, "A manifest part is missing or has a different ETag");
            long size = row.getLong(2); if (index + 1 < manifest.size() && size < MIN_PART_SIZE) throw error("EntityTooSmall", 400, "All but the final part must be at least 5 MiB");
            String etag = row.getString(3); multipart.update(HexFormat.of().parseHex(etag)); files.add(row.getString(1)); Files.copy(payload(row.getString(1), staging), out);
          }
        }
      }
      Staged staged = stage(Files.newInputStream(joined)); delete(joined);
      String composite = hex(multipart.digest()) + "-" + manifest.size();
      staged = new Staged(staged.path, staged.size, composite, staged.sha256);
      c.commit();
      c.setAutoCommit(true);
      PutObjectRequest request = PutObjectRequest.builder().bucket(upload.bucket).key(upload.key).contentType(upload.type).metadata(upload.metadata).build();
      PutObjectResponse response = publish(request, staged);
      files.clear();
      try (PreparedStatement all = c.prepareStatement("SELECT file FROM parts WHERE upload_id=?")) {
        all.setString(1, m.id());
        try (ResultSet rows = all.executeQuery()) { while (rows.next()) files.add(rows.getString(1)); }
      }
      try (Connection cleanup = connection(); PreparedStatement d = cleanup.prepareStatement("DELETE FROM parts WHERE upload_id=?"); PreparedStatement u = cleanup.prepareStatement("DELETE FROM uploads WHERE id=?")) {
        cleanup.setAutoCommit(false); d.setString(1, m.id()); d.executeUpdate(); u.setString(1, m.id()); u.executeUpdate(); cleanup.commit();
      }
      for (String file : files) delete(payload(file, staging)); return SdkResponses.completeResponse(response.eTag());
    } catch (S3Exception e) { delete(joined); throw e;
    } catch (Exception e) { delete(joined); throw new IllegalStateException("multipart completion failed", e);
    } finally { lock.unlock(); }
  }
  @Override public void abortMultipartUpload(MultipartUpload upload) {
    ReentrantLock guard = lock(upload.containerName(), upload.blobName());
    guard.lock();
    try {
      List<String> files = new ArrayList<>();
      try (Connection c = connection()) {
        c.setAutoCommit(false);
        upload(c, upload);
        try (PreparedStatement query = c.prepareStatement("SELECT file FROM parts WHERE upload_id=?")) {
          query.setString(1, upload.id());
          try (ResultSet rows = query.executeQuery()) { while (rows.next()) files.add(rows.getString(1)); }
        }
        try (PreparedStatement delete = c.prepareStatement("DELETE FROM parts WHERE upload_id=?")) { delete.setString(1, upload.id()); delete.executeUpdate(); }
        try (PreparedStatement delete = c.prepareStatement("DELETE FROM uploads WHERE id=?")) { delete.setString(1, upload.id()); delete.executeUpdate(); }
        c.commit();
      } catch (SQLException error) { throw new IllegalStateException(error); }
      for (String file : files) delete(payload(file, staging));
    } finally { guard.unlock(); }
  }
  @Override public long getMinimumMultipartPartSize() { return MIN_PART_SIZE; }
  @Override public List<software.amazon.awssdk.services.s3.model.MultipartUpload> listMultipartUploads(String bucket) {
    List<software.amazon.awssdk.services.s3.model.MultipartUpload> uploads = new ArrayList<>();
    try (Connection c = connection(); PreparedStatement query = c.prepareStatement("SELECT id,key,created FROM uploads WHERE bucket=? ORDER BY key,id")) {
      query.setString(1, bucket);
      try (ResultSet rows = query.executeQuery()) {
        while (rows.next()) uploads.add(software.amazon.awssdk.services.s3.model.MultipartUpload.builder()
            .uploadId(rows.getString(1)).key(rows.getString(2)).initiated(Instant.ofEpochMilli(rows.getLong(3))).build());
      }
    } catch (SQLException error) { throw new IllegalStateException(error); }
    return uploads;
  }
  @Override public void close() {}
  void checkpoint() { try (Connection c=connection(); Statement s=c.createStatement()) { s.execute("PRAGMA wal_checkpoint(FULL)"); } catch(SQLException e){throw new IllegalStateException(e);} }
}
