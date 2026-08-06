import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_document_parser_module():
    module_name = "tests._document_parser_under_test"
    module_path = Path(__file__).resolve().parents[1] / "ragtime/indexer/document_parser.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load document_parser module for tests")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    fake_file_constants = types.SimpleNamespace(
        ANYDOC_DOCUMENT_EXTENSIONS={".pdf", ".docx", ".csv", ".docm", ".xlsm", ".xlsb", ".pps", ".pot", ".pptm", ".ppsx", ".ppsm"},
        DOCUMENT_EXTENSIONS={".txt", ".pdf", ".docx", ".eml"},
        OCR_EXTENSIONS={".png", ".jpg"},
        RAW_CAMERA_EXTENSIONS=set(),
    )
    fake_vision_models = types.SimpleNamespace(
        VisionOcrResult=object,
        extract_text_with_vision=mock.AsyncMock(return_value=""),
        extract_text_with_vision_structured=mock.AsyncMock(return_value=None),
    )
    fake_logger = mock.Mock()
    fake_logging_module = types.SimpleNamespace(get_logger=mock.Mock(return_value=fake_logger))

    with mock.patch.dict(
        sys.modules,
        {
            "ragtime.core.file_constants": fake_file_constants,
            "ragtime.core.logging": fake_logging_module,
            "ragtime.core.vision_models": fake_vision_models,
        },
        clear=False,
    ):
        spec.loader.exec_module(module)
    return module


class DocumentConversionTests(unittest.TestCase):
    def _fake_anydoc(self, **overrides):
        class ConvertError(Exception):
            pass

        class UnsupportedError(ConvertError):
            pass

        class MalformedError(ConvertError):
            pass

        class EncryptedError(ConvertError):
            pass

        class ResourceLimitError(ConvertError):
            pass

        class MissingPartError(ConvertError):
            pass

        module = types.SimpleNamespace(
            ConvertError=ConvertError,
            UnsupportedError=UnsupportedError,
            MalformedError=MalformedError,
            EncryptedError=EncryptedError,
            ResourceLimitError=ResourceLimitError,
            MissingPartError=MissingPartError,
            format_from_bytes=mock.Mock(return_value="pdf"),
            format_from_extension=mock.Mock(return_value=None),
            to_markdown_bytes=mock.Mock(return_value=b"converted text"),
        )
        for name, value in overrides.items():
            setattr(module, name, value)
        return module

    def test_convert_document_bytes_prefers_detected_content_type(self) -> None:
        from ragtime.core.document_conversion import convert_document_bytes

        fake_anydoc = self._fake_anydoc(
            format_from_bytes=mock.Mock(return_value="pdf"),
            format_from_extension=mock.Mock(return_value="docx"),
        )

        with mock.patch.dict(sys.modules, {"anydoc": fake_anydoc}, clear=False):
            result = convert_document_bytes(b"%PDF-1.7", ".docx")

        self.assertEqual(result.text, "converted text")
        self.assertIsNone(result.failure)
        fake_anydoc.format_from_bytes.assert_called_once_with(b"%PDF-1.7")
        fake_anydoc.format_from_extension.assert_not_called()
        fake_anydoc.to_markdown_bytes.assert_called_once_with(b"%PDF-1.7", "pdf")

    def test_convert_document_bytes_falls_back_to_extension(self) -> None:
        from ragtime.core.document_conversion import convert_document_bytes

        fake_anydoc = self._fake_anydoc(
            format_from_bytes=mock.Mock(return_value=None),
            format_from_extension=mock.Mock(return_value="csv"),
            to_markdown_bytes=mock.Mock(return_value=b"a,b\n1,2\n"),
        )

        with mock.patch.dict(sys.modules, {"anydoc": fake_anydoc}, clear=False):
            result = convert_document_bytes(b"a,b\n1,2\n", ".csv")

        self.assertEqual(result.text, "a,b\n1,2\n")
        self.assertIsNone(result.failure)
        fake_anydoc.format_from_bytes.assert_called_once_with(b"a,b\n1,2\n")
        fake_anydoc.format_from_extension.assert_called_once_with(".csv")
        fake_anydoc.to_markdown_bytes.assert_called_once_with(b"a,b\n1,2\n", "csv")

    def test_convert_document_bytes_returns_typed_failures(self) -> None:
        from ragtime.core.document_conversion import DocumentConversionFailure, convert_document_bytes

        cases = [
            ("UnsupportedError", DocumentConversionFailure.UNSUPPORTED),
            ("MalformedError", DocumentConversionFailure.MALFORMED),
            ("EncryptedError", DocumentConversionFailure.ENCRYPTED),
            ("ResourceLimitError", DocumentConversionFailure.RESOURCE_LIMIT),
            ("MissingPartError", DocumentConversionFailure.MISSING_PART),
            ("RuntimeError", DocumentConversionFailure.UNEXPECTED),
        ]

        for error_name, expected in cases:
            with self.subTest(error_name=error_name):
                fake_anydoc = self._fake_anydoc()
                error_type = getattr(fake_anydoc, error_name, type(error_name, (Exception,), {}))
                fake_anydoc.to_markdown_bytes = mock.Mock(side_effect=error_type("boom"))

                with mock.patch.dict(sys.modules, {"anydoc": fake_anydoc}, clear=False):
                    result = convert_document_bytes(b"data", ".pdf")

                self.assertEqual(result.failure, expected)
                self.assertEqual(result.detail, "boom")
                self.assertEqual(result.text, "")

    def test_convert_document_bytes_reports_dependency_failures(self) -> None:
        from ragtime.core.document_conversion import DocumentConversionFailure, convert_document_bytes

        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "anydoc":
                raise ImportError("missing anydoc")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = convert_document_bytes(b"data", ".pdf")

        self.assertEqual(result.failure, DocumentConversionFailure.DEPENDENCY)
        self.assertEqual(result.detail, "missing anydoc")
        self.assertEqual(result.text, "")


class DocumentParserRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_text_from_file_uses_document_conversion_adapter(self) -> None:
        document_parser = _load_document_parser_module()

        with mock.patch.object(
            document_parser,
            "convert_document_bytes",
            return_value=types.SimpleNamespace(text="converted pdf", failure=None, detail=None),
        ) as convert_mock:
            result = document_parser.extract_text_from_file(Path("example.pdf"), content=b"pdf bytes")

        self.assertEqual(result, "converted pdf")
        convert_mock.assert_called_once_with(b"pdf bytes", ".pdf")

    async def test_extract_text_from_file_async_uses_document_conversion_adapter(self) -> None:
        document_parser = _load_document_parser_module()

        with mock.patch.object(
            document_parser,
            "convert_document_bytes",
            return_value=types.SimpleNamespace(text="converted docx", failure=None, detail=None),
        ) as convert_mock:
            result = await document_parser.extract_text_from_file_async(Path("example.docx"), content=b"docx bytes")

        self.assertEqual(result, "converted docx")
        convert_mock.assert_called_once_with(b"docx bytes", ".docx")

    def test_extract_text_from_file_routes_csv_through_document_conversion_adapter(self) -> None:
        document_parser = _load_document_parser_module()

        with mock.patch.object(
            document_parser,
            "convert_document_bytes",
            return_value=types.SimpleNamespace(text="converted csv", failure=None, detail=None),
        ) as convert_mock:
            result = document_parser.extract_text_from_file(Path("table.csv"), content=b"a,b\n1,2\n")

        self.assertEqual(result, "converted csv")
        convert_mock.assert_called_once_with(b"a,b\n1,2\n", ".csv")

    def test_extract_text_from_file_keeps_eml_routing(self) -> None:
        document_parser = _load_document_parser_module()

        with (
            mock.patch.object(
                document_parser,
                "convert_document_bytes",
                return_value=types.SimpleNamespace(text="wrong", failure=None, detail=None),
            ) as convert_mock,
            mock.patch.object(
                document_parser,
                "_extract_eml",
                return_value="mail body",
            ) as eml_mock,
        ):
            result = document_parser.extract_text_from_file(Path("message.eml"), content=b"mail bytes")

        self.assertEqual(result, "mail body")
        eml_mock.assert_called_once_with(b"mail bytes")
        convert_mock.assert_not_called()

    def test_extract_text_from_file_keeps_msg_routing(self) -> None:
        document_parser = _load_document_parser_module()

        with (
            mock.patch.object(
                document_parser,
                "convert_document_bytes",
                return_value=types.SimpleNamespace(text="wrong", failure=None, detail=None),
            ) as convert_mock,
            mock.patch.object(
                document_parser,
                "_extract_msg",
                return_value="msg body",
            ) as msg_mock,
        ):
            result = document_parser.extract_text_from_file(Path("message.msg"), content=b"msg bytes")

        self.assertEqual(result, "msg body")
        msg_mock.assert_called_once_with(b"msg bytes")
        convert_mock.assert_not_called()
