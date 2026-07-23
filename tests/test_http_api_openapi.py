import unittest

from ragtime.http_api.models import OpenApiCatalog
from ragtime.http_api.openapi import normalize_openapi_source, search_openapi_catalog


class HttpApiOpenApiTests(unittest.TestCase):
    def test_normalize_openapi_source_json_returns_catalog_and_metadata(self) -> None:
        catalog, metadata = normalize_openapi_source(
            spec_url="https://api.example.com/openapi.json",
            document='{"openapi":"3.1.0","info":{"title":"Demo","version":"1.0"},"paths":{"/items":{"get":{"operationId":"listItems","summary":"List items","tags":["items"]}}}}',
            document_name="openapi.json",
        )

        self.assertIsInstance(catalog, OpenApiCatalog)
        self.assertEqual(catalog.title, "Demo")
        self.assertEqual(catalog.version, "1.0")
        self.assertEqual(len(catalog.operations), 1)
        self.assertEqual(catalog.operations[0].method, "GET")
        self.assertEqual(catalog.operations[0].path, "/items")
        self.assertEqual(metadata["openapi_source_url"], "https://api.example.com/openapi.json")
        self.assertEqual(metadata["openapi_source_name"], "openapi.json")
        self.assertTrue(metadata["openapi_source_hash"])

    def test_normalize_openapi_source_yaml_supports_safe_loader(self) -> None:
        catalog, metadata = normalize_openapi_source(
            document="""
openapi: 3.1.0
info:
  title: Demo YAML
  version: '1.2'
paths:
  /users:
    post:
      operationId: createUser
      summary: Create user
""",
            document_name="openapi.yaml",
        )

        self.assertEqual(catalog.title, "Demo YAML")
        self.assertEqual(catalog.operations[0].method, "POST")
        self.assertEqual(metadata["openapi_source_name"], "openapi.yaml")

    def test_normalize_openapi_source_rejects_remote_refs(self) -> None:
        with self.assertRaises(ValueError):
            normalize_openapi_source(
                document='{"openapi":"3.1.0","info":{"title":"Demo","version":"1.0"},"paths":{"/items":{"get":{"responses":{"200":{"$ref":"https://evil.example/ref.json"}}}}}}'
            )

    def test_search_openapi_catalog_matches_metadata(self) -> None:
        catalog, _metadata = normalize_openapi_source(
            document='{"openapi":"3.1.0","info":{"title":"Demo","version":"1.0"},"paths":{"/items":{"get":{"operationId":"listItems","summary":"List items"}},"/users":{"post":{"operationId":"createUser","summary":"Create user"}}}}'
        )

        matches = search_openapi_catalog(catalog, "create user")
        self.assertEqual([item.operation_id for item in matches], ["createUser"])

    def test_normalize_openapi_source_rejects_oversized_document(self) -> None:
        large = "{" + '"openapi":"3.1.0","info":{"title":"X","version":"1"},"paths":{}' + (" " * (2 * 1024 * 1024)) + "}"
        with self.assertRaises(ValueError):
            normalize_openapi_source(document=large)

    def test_normalize_openapi_source_rejects_deep_document(self) -> None:
        nested: dict[str, object] = {}
        current = nested
        for _ in range(41):
            child: dict[str, object] = {}
            current["x"] = child
            current = child
        current["openapi"] = "3.1.0"
        with self.assertRaises(ValueError):
            normalize_openapi_source(document=str(nested).replace("'", '"'))

    def test_normalize_openapi_source_rejects_more_than_500_operations(self) -> None:
        paths = {f"/items/{idx}": {"get": {"operationId": f"getItem{idx}"}} for idx in range(501)}
        import json

        with self.assertRaises(ValueError):
            normalize_openapi_source(document=json.dumps({"openapi": "3.1.0", "info": {"title": "Demo", "version": "1"}, "paths": paths}))

    def test_normalize_openapi_source_rejects_yaml_aliases(self) -> None:
        with self.assertRaises(ValueError):
            normalize_openapi_source(
                document="""
openapi: 3.1.0
info: &info
  title: Demo
  version: '1'
paths:
  /items:
    get:
      <<: *info
      operationId: listItems
"""
            )


if __name__ == "__main__":
    unittest.main()
