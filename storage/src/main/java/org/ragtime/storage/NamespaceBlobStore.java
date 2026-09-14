package org.ragtime.storage;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.gaul.s3proxy.blobstore.BlobStore;
import org.gaul.s3proxy.blobstore.domain.MultipartUpload;
import org.gaul.s3proxy.awssdk.AwsS3SdkBlobStore;
import software.amazon.awssdk.awscore.exception.AwsErrorDetails;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.services.s3.model.*;

/** Tenant facade. Unimplemented SPI operations use denying defaults, never a raw store. */
final class NamespaceBlobStore implements BlobStore {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final StorageEngine engine;
    private final String workspaceId;
    private final byte[] tokenKey;

    NamespaceBlobStore(StorageEngine engine, String workspaceId, String key, BlobStore unused) {
        this.engine = engine;
        this.workspaceId = workspaceId;
        this.tokenKey = key.getBytes(StandardCharsets.UTF_8);
    }

    private JsonNode bucket(String name) {
        JsonNode workspace = engine.registry().workspace(workspaceId);
        if (workspace != null && !"revoked".equals(workspace.path("state").asText())) {
            for (JsonNode bucket : workspace.path("buckets")) {
                if (name.equals(bucket.path("name").asText())) return bucket;
            }
        }
        throw error(404, "NoSuchBucket");
    }

    private String physicalBucket(String name) {
        return engine.physicalBucket(bucket(name).path("backend_id").asText());
    }

    private BlobStore store(String name) {
        return engine.physicalStore(bucket(name).path("backend_id").asText());
    }

    private String prefix(String name) {
        return engine.physicalPrefix(workspaceId, bucket(name).path("id").asText());
    }

    private String physicalKey(String name, String key) {
        return prefix(name) + key;
    }

    private static S3Exception error(int status, String code) {
        return (S3Exception) S3Exception.builder().statusCode(status).message(code)
            .awsErrorDetails(AwsErrorDetails.builder().errorCode(code).errorMessage(code).build()).build();
    }

    @Override public ListBucketsResponse list() {
        JsonNode workspace = engine.registry().workspace(workspaceId);
        List<Bucket> buckets = new ArrayList<>();
        if (workspace != null && !"revoked".equals(workspace.path("state").asText())) {
            for (JsonNode b : workspace.path("buckets")) buckets.add(Bucket.builder().name(b.path("name").asText()).build());
        }
        return ListBucketsResponse.builder().buckets(buckets).build();
    }

    @Override public HeadBucketResponse headBucket(HeadBucketRequest request) {
        bucket(request.bucket());
        return HeadBucketResponse.builder().build();
    }

    @Override public CreateBucketResponse createContainer(CreateBucketRequest request) { throw error(403, "AccessDenied"); }
    @Override public void deleteBucket(String name) { throw error(403, "AccessDenied"); }
    @Override public void deleteContainer(String name) { throw error(403, "AccessDenied"); }
    @Override public void clearContainer(ListObjectsV2Request request) { throw error(403, "AccessDenied"); }
    @Override public BucketCannedACL getContainerAccess(String name) { bucket(name); return BucketCannedACL.PRIVATE; }
    @Override public void setContainerAccess(String name, BucketCannedACL acl) { throw error(403, "AccessDenied"); }
    @Override public ObjectCannedACL getBlobAccess(String name, String key) { bucket(name); return ObjectCannedACL.PRIVATE; }
    @Override public void setBlobAccess(String name, String key, ObjectCannedACL acl) { throw error(403, "AccessDenied"); }

    @Override public boolean blobExists(String name, String key) {
        return store(name).blobExists(physicalBucket(name), physicalKey(name, key));
    }

    @Override public PutObjectResponse putBlob(PutObjectRequest request, InputStream payload) {
        return engine.mutate(workspaceId, () -> {
            try {
                BlobStore target = store(request.bucket());
                PutObjectRequest mapped = request.toBuilder().bucket(physicalBucket(request.bucket()))
                    .key(physicalKey(request.bucket(), request.key())).build();
                return target.putBlob(mapped, payload);
            } catch (S3Exception exception) {
                org.slf4j.LoggerFactory.getLogger(NamespaceBlobStore.class).warn("Object write refused: status={} code={}",
                    exception.statusCode(), exception.awsErrorDetails() == null ? "unknown" : exception.awsErrorDetails().errorCode());
                throw exception;
            }
        });
    }

    @Override public ResponseInputStream<GetObjectResponse> getBlob(GetObjectRequest request) {
        return store(request.bucket()).getBlob(request.toBuilder().bucket(physicalBucket(request.bucket()))
            .key(physicalKey(request.bucket(), request.key())).build());
    }

    @Override public HeadObjectResponse blobMetadata(HeadObjectRequest request) {
        return store(request.bucket()).blobMetadata(request.toBuilder().bucket(physicalBucket(request.bucket()))
            .key(physicalKey(request.bucket(), request.key())).build());
    }

    @Override public DeleteObjectResponse removeBlob(DeleteObjectRequest request) {
        return engine.mutate(workspaceId, () -> store(request.bucket()).removeBlob(request.toBuilder()
            .bucket(physicalBucket(request.bucket())).key(physicalKey(request.bucket(), request.key())).build()));
    }

    @Override public CopyObjectResponse copyBlob(CopyObjectRequest request) {
        return engine.mutate(workspaceId, () -> {
            JsonNode source = bucket(request.sourceBucket()), destination = bucket(request.destinationBucket());
            if (!source.path("backend_id").equals(destination.path("backend_id"))) throw error(501, "NotImplemented");
            return store(request.sourceBucket()).copyBlob(request.toBuilder()
                .sourceBucket(physicalBucket(request.sourceBucket())).sourceKey(physicalKey(request.sourceBucket(), request.sourceKey()))
                .destinationBucket(physicalBucket(request.destinationBucket())).destinationKey(physicalKey(request.destinationBucket(), request.destinationKey())).build());
        });
    }

    @Override public ListObjectsV2Response list(ListObjectsV2Request request) {
        String name = request.bucket(), scope = prefix(name);
        String logicalPrefix = request.prefix() == null ? "" : request.prefix();
        String delimiter = request.delimiter() == null ? "" : request.delimiter();
        String continuation = request.continuationToken() == null ? null
            : decode(request.continuationToken(), "page", name, logicalPrefix, delimiter);
        ListObjectsV2Request mapped = request.toBuilder().bucket(physicalBucket(name)).prefix(scope + logicalPrefix)
            .continuationToken(continuation).startAfter(request.startAfter() == null ? null : scope + request.startAfter()).build();
        ListObjectsV2Response raw = store(name).list(mapped);
        List<S3Object> objects = new ArrayList<>();
        for (S3Object object : raw.contents()) {
            if (!object.key().startsWith(scope)) throw error(502, "InvalidBackendResponse");
            objects.add(object.toBuilder().key(object.key().substring(scope.length())).build());
        }
        List<CommonPrefix> prefixes = new ArrayList<>();
        for (CommonPrefix item : raw.commonPrefixes()) {
            if (!item.prefix().startsWith(scope)) throw error(502, "InvalidBackendResponse");
            prefixes.add(item.toBuilder().prefix(item.prefix().substring(scope.length())).build());
        }
        return raw.toBuilder().name(name).prefix(logicalPrefix).delimiter(request.delimiter())
            .startAfter(request.startAfter()).continuationToken(request.continuationToken())
            .contents(objects).commonPrefixes(prefixes).keyCount(objects.size() + prefixes.size())
            .nextContinuationToken(raw.nextContinuationToken() == null ? null
                : encode("page", name, logicalPrefix, delimiter, raw.nextContinuationToken())).build();
    }

    @Override public MultipartUpload initiateMultipartUpload(CreateMultipartUploadRequest request) {
        return engine.mutate(workspaceId, () -> {
            MultipartUpload raw = store(request.bucket()).initiateMultipartUpload(request.toBuilder()
                .bucket(physicalBucket(request.bucket())).key(physicalKey(request.bucket(), request.key())).build());
            String id = encode("multipart", request.bucket(), request.key(), "", raw.id());
            return new MultipartUpload(id, request, raw.response());
        });
    }

    private MultipartUpload physical(MultipartUpload upload) {
        String id = decode(upload.id(), "multipart", upload.containerName(), upload.blobName(), "");
        return new MultipartUpload(id, upload.request().toBuilder().bucket(physicalBucket(upload.containerName()))
            .key(physicalKey(upload.containerName(), upload.blobName())).build(), upload.response());
    }

    @Override public UploadPartResponse uploadMultipartPart(MultipartUpload upload, UploadPartRequest request, InputStream stream) {
        return engine.mutate(workspaceId, () -> {
            MultipartUpload raw = physical(upload);
            BlobStore target = store(upload.containerName());
            UploadPartRequest mapped = request.toBuilder().bucket(raw.containerName())
                .key(raw.blobName()).uploadId(raw.id()).build();
            return target.uploadMultipartPart(raw, mapped, stream);
        });
    }

    @Override public CompleteMultipartUploadResponse completeMultipartUpload(MultipartUpload upload, CompleteMultipartUploadRequest request) {
        return engine.mutate(workspaceId, () -> {
            MultipartUpload raw = physical(upload);
            return store(upload.containerName()).completeMultipartUpload(raw, request.toBuilder().bucket(raw.containerName())
                .key(raw.blobName()).uploadId(raw.id()).build());
        });
    }

    @Override public void abortMultipartUpload(MultipartUpload upload) {
        engine.mutate(workspaceId, () -> { store(upload.containerName()).abortMultipartUpload(physical(upload)); return null; });
    }

    @Override public List<Part> listMultipartUpload(MultipartUpload upload) {
        return store(upload.containerName()).listMultipartUpload(physical(upload));
    }

    @Override public List<software.amazon.awssdk.services.s3.model.MultipartUpload> listMultipartUploads(String name) {
        String scope = prefix(name);
        List<software.amazon.awssdk.services.s3.model.MultipartUpload> result = new ArrayList<>();
        for (var upload : store(name).listMultipartUploads(physicalBucket(name))) {
            if (!upload.key().startsWith(scope)) continue;
            String key = upload.key().substring(scope.length());
            result.add(upload.toBuilder().key(key).uploadId(encode("multipart", name, key, "", upload.uploadId())).build());
        }
        return result;
    }

    @Override public long getMinimumMultipartPartSize() { return 5L * 1024 * 1024; }
    @Override public void close() { /* Engine owns the shared physical connections. */ }

    private String encode(String kind, String name, String key, String delimiter, String upstream) {
        JsonNode b = bucket(name);
        try {
            byte[] payload = JSON.writeValueAsBytes(List.of(kind, workspaceId, b.path("id").asText(),
                b.path("backend_id").asText(), key, delimiter, upstream));
            var encoder = Base64.getUrlEncoder().withoutPadding();
            return encoder.encodeToString(payload) + "." + encoder.encodeToString(sign(payload));
        } catch (Exception ex) { throw error(500, "InternalError"); }
    }

    private String decode(String token, String kind, String name, String key, String delimiter) {
        JsonNode b = bucket(name);
        try {
            if (token.length() > 32768) throw new IllegalArgumentException();
            String[] parts = token.split("\\.", -1);
            if (parts.length != 2) throw new IllegalArgumentException();
            byte[] payload = Base64.getUrlDecoder().decode(parts[0]);
            if (!MessageDigest.isEqual(sign(payload), Base64.getUrlDecoder().decode(parts[1]))) throw new IllegalArgumentException();
            JsonNode values = JSON.readTree(payload);
            List<String> expected = List.of(kind, workspaceId, b.path("id").asText(), b.path("backend_id").asText(), key, delimiter);
            if (!values.isArray() || values.size() != 7) throw new IllegalArgumentException();
            for (int i = 0; i < expected.size(); i++) {
                if (!values.get(i).isTextual() || !expected.get(i).equals(values.get(i).asText())) throw new IllegalArgumentException();
            }
            if (!values.get(6).isTextual()) throw new IllegalArgumentException();
            return values.get(6).asText();
        } catch (Exception ex) { throw error(400, "multipart".equals(kind) ? "NoSuchUpload" : "InvalidArgument"); }
    }

    private byte[] sign(byte[] payload) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(tokenKey, "HmacSHA256"));
        mac.update("ragtime-s3-token-v1\0".getBytes(StandardCharsets.UTF_8));
        return mac.doFinal(payload);
    }
}
