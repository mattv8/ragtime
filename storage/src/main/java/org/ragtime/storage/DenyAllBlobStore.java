package org.ragtime.storage;

import java.io.InputStream;
import java.util.List;
import org.gaul.s3proxy.blobstore.BlobStore;
import org.gaul.s3proxy.blobstore.ForwardingBlobStore;
import org.gaul.s3proxy.blobstore.domain.MultipartUpload;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.services.s3.model.*;

/** Builder fallback only: dynamic locator grants are the sole path to tenant data. */
final class DenyAllBlobStore extends ForwardingBlobStore {
  DenyAllBlobStore(BlobStore delegate) { super(delegate); }
  private static SecurityException deny(){ return new SecurityException("access denied"); }
  @Override public ListBucketsResponse list(){throw deny();}
  @Override public ListObjectsV2Response list(ListObjectsV2Request r){throw deny();}
  @Override public HeadBucketResponse headBucket(HeadBucketRequest r){throw deny();}
  @Override public CreateBucketResponse createContainer(CreateBucketRequest r){throw deny();}
  @Override public void deleteBucket(String bucket){throw deny();}
  @Override public boolean blobExists(String bucket,String key){throw deny();}
  @Override public PutObjectResponse putBlob(PutObjectRequest r,InputStream in){throw deny();}
  @Override public CopyObjectResponse copyBlob(CopyObjectRequest r){throw deny();}
  @Override public HeadObjectResponse blobMetadata(HeadObjectRequest r){throw deny();}
  @Override public ResponseInputStream<GetObjectResponse> getBlob(GetObjectRequest r){throw deny();}
  @Override public DeleteObjectResponse removeBlob(DeleteObjectRequest r){throw deny();}
  @Override public MultipartUpload initiateMultipartUpload(CreateMultipartUploadRequest r){throw deny();}
  @Override public void abortMultipartUpload(MultipartUpload m){throw deny();}
  @Override public CompleteMultipartUploadResponse completeMultipartUpload(MultipartUpload m,CompleteMultipartUploadRequest r){throw deny();}
  @Override public UploadPartResponse uploadMultipartPart(MultipartUpload m,UploadPartRequest r,InputStream in){throw deny();}
  @Override public List<Part> listMultipartUpload(MultipartUpload m){throw deny();}
}
