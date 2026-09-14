package org.ragtime.storage;

import static org.junit.jupiter.api.Assertions.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class RegistryTest {
  @Test void encryptedStatePersistsAndWrongKeyFailsClosed() throws Exception {
    Path root=Files.createTempDirectory("registry");
    try (Registry first=new Registry(root," correct ")) {
      first.mutate(s -> { var w=s.withObject("workspaces").putObject("ws"); w.put("workspace_id","ws"); w.put("access_key_id","key"); w.put("secret_access_key","secret"); w.put("state","ready"); });
    }
    try (Registry reopened=new Registry(root,"correct")) { assertEquals("ws",reopened.findIdentity("key").path("workspace_id").asText()); }
    assertThrows(IOException.class, () -> new Registry(root,"wrong"));
  }
  @Test void identityCollisionsCannotPublish() throws Exception {
    try (Registry registry=new Registry(Files.createTempDirectory("registry"),"key")) {
      registry.mutate(s -> { for(String id : new String[]{"one","two"}) { var w=s.withObject("workspaces").putObject(id); w.put("workspace_id",id); w.put("access_key_id","same"); w.put("state","ready"); } });
      fail("collision should have failed");
    } catch (IllegalArgumentException expected) { assertTrue(expected.getMessage().contains("duplicate")); }
  }
}
