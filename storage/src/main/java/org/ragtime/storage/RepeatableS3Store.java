package org.ragtime.storage;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.gaul.s3proxy.blobstore.BlobStore;
import org.gaul.s3proxy.blobstore.ForwardingBlobStore;
import org.gaul.s3proxy.blobstore.domain.MultipartUpload;
import software.amazon.awssdk.services.s3.model.*;

/** Shared external-write adapter for SDK traffic, legacy imports, and migration workers. */
final class RepeatableS3Store extends ForwardingBlobStore {
    RepeatableS3Store(BlobStore store) { super(store); }

    @Override public PutObjectResponse putBlob(PutObjectRequest request, InputStream input) {
        try (InputStream body = repeatable(input, request.contentLength())) {
            return delegate().putBlob(request, body);
        } catch (IOException error) { throw new IllegalStateException("Unable to stage external object", error); }
    }

    @Override public UploadPartResponse uploadMultipartPart(MultipartUpload upload, UploadPartRequest request, InputStream input) {
        try (InputStream body = repeatable(input, request.contentLength())) {
            return delegate().uploadMultipartPart(upload, request, body);
        } catch (IOException error) { throw new IllegalStateException("Unable to stage external multipart part", error); }
    }

    private InputStream repeatable(InputStream input, Long length) throws IOException {
        if (input.markSupported()) return input;
        if (length == null || length < 0) throw new IOException("External payload needs a known content length");
        return new DiskBackedInputStream(input, length);
    }

    private static final class DiskBackedInputStream extends InputStream {
        private final Path file;
        private final FileChannel channel;
        private long mark;

        DiskBackedInputStream(InputStream source, long length) throws IOException {
            file = Files.createTempFile("ragtime-s3-", ".payload");
            try (source) {
                Files.copy(source, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                if (Files.size(file) != length) throw new IOException("Payload length changed while spooling");
                channel = FileChannel.open(file, StandardOpenOption.READ);
            } catch (IOException | RuntimeException error) {
                Files.deleteIfExists(file);
                throw error;
            }
        }

        @Override public int read() throws IOException {
            ByteBuffer buffer = ByteBuffer.allocate(1);
            return channel.read(buffer) == -1 ? -1 : Byte.toUnsignedInt(buffer.array()[0]);
        }
        @Override public int read(byte[] bytes, int offset, int length) throws IOException { return channel.read(ByteBuffer.wrap(bytes, offset, length)); }
        @Override public boolean markSupported() { return true; }
        @Override public void mark(int readLimit) { try { mark = channel.position(); } catch (IOException error) { throw new IllegalStateException(error); } }
        @Override public void reset() throws IOException { channel.position(mark); }
        @Override public void close() throws IOException { try { channel.close(); } finally { Files.deleteIfExists(file); } }
    }
}
