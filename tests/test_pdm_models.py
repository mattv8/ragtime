import unittest

from ragtime.indexer.models import (
    PdmBomComponentModel,
    PdmConfigurationStateModel,
    PdmDocumentStateModel,
    PdmPropertyValueModel,
)


def _value(
    variable_id: int,
    variable_name: str,
    value_text: str,
    origin_revision: int = 1,
    is_blank: bool = False,
) -> PdmPropertyValueModel:
    return PdmPropertyValueModel(
        variable_id=variable_id,
        variable_name=variable_name,
        value_text=value_text,
        is_blank=is_blank,
        origin_revision=origin_revision,
    )


class PdmDocumentStateModelTests(unittest.TestCase):
    def _document(self, **changes: object) -> PdmDocumentStateModel:
        data = {
            "document_id": 10,
            "filename": "sample.SLDPRT",
            "target_revision": 2,
            "document_values": {"Description": _value(2, "Description", "Base description", 2)},
        }
        data.update(changes)
        return PdmDocumentStateModel(**data)

    def test_embedding_text_includes_distinct_configuration_values(self) -> None:
        document = self._document(
            configurations=[
                PdmConfigurationStateModel(
                    configuration_id=2,
                    name="Large",
                    values={"Description": _value(2, "Description", "Large description")},
                ),
                PdmConfigurationStateModel(
                    configuration_id=1,
                    name="Small",
                    values={"Description": _value(2, "Description", "Small description")},
                ),
            ]
        )

        text = document.to_embedding_text()

        self.assertIn("- Small", text)
        self.assertIn("  - Description: Small description", text)
        self.assertIn("- Large", text)
        self.assertIn("  - Description: Large description", text)

    def test_hash_changes_when_value_is_cleared(self) -> None:
        before = self._document(document_values={"Description": _value(2, "Description", "Present", 2)})
        after = self._document(document_values={"Description": _value(2, "Description", "", 2, is_blank=True)})

        self.assertNotEqual(before.compute_metadata_hash(), after.compute_metadata_hash())

    def test_hash_changes_when_origin_revision_changes(self) -> None:
        before = self._document(document_values={"Description": _value(2, "Description", "Present", 1)})
        after = self._document(document_values={"Description": _value(2, "Description", "Present", 2)})

        self.assertNotEqual(before.compute_metadata_hash(), after.compute_metadata_hash())

    def test_hash_changes_when_configuration_is_removed(self) -> None:
        configuration = PdmConfigurationStateModel(
            configuration_id=1,
            name="Default",
            values={"Description": _value(2, "Description", "Configured")},
        )

        self.assertNotEqual(
            self._document(configurations=[configuration]).compute_metadata_hash(),
            self._document(configurations=[]).compute_metadata_hash(),
        )

    def test_hash_is_independent_of_input_order(self) -> None:
        first = self._document(
            folder_paths=["/vault/B", "/vault/A"],
            document_values={
                "Material": _value(3, "Material", "Steel"),
                "Description": _value(2, "Description", "Widget"),
            },
            configurations=[
                PdmConfigurationStateModel(
                    configuration_id=2,
                    name="Large",
                    values={"Description": _value(2, "Description", "Large")},
                ),
                PdmConfigurationStateModel(
                    configuration_id=1,
                    name="Small",
                    values={"Description": _value(2, "Description", "Small")},
                ),
            ],
        )
        second = self._document(
            folder_paths=["/vault/A", "/vault/B"],
            document_values={
                "Description": _value(2, "Description", "Widget"),
                "Material": _value(3, "Material", "Steel"),
            },
            configurations=[
                PdmConfigurationStateModel(
                    configuration_id=1,
                    name="Small",
                    values={"Description": _value(2, "Description", "Small")},
                ),
                PdmConfigurationStateModel(
                    configuration_id=2,
                    name="Large",
                    values={"Description": _value(2, "Description", "Large")},
                ),
            ],
        )

        self.assertEqual(first.compute_metadata_hash(), second.compute_metadata_hash())

    def test_blank_values_are_excluded_from_text_but_preserved_in_dump(self) -> None:
        document = self._document(document_values={"Description": _value(2, "Description", "", 2, is_blank=True)})

        self.assertNotIn("Description:", document.to_embedding_text())
        self.assertEqual(document.model_dump()["document_values"]["Description"]["value_text"], "")
        self.assertTrue(document.model_dump()["document_values"]["Description"]["is_blank"])

    def test_bom_component_without_quantity_omits_quantity_text(self) -> None:
        document = self._document(bom_components=[PdmBomComponentModel(document_id=20, filename="child.SLDPRT", configuration="Default")])

        text = document.to_embedding_text()

        self.assertIn("1. child.SLDPRT (Config: Default)", text)
        self.assertNotIn("Qty", text)

    def test_folder_path_order_does_not_affect_hash(self) -> None:
        self.assertEqual(
            self._document(folder_paths=["/vault/B", "/vault/A"]).compute_metadata_hash(),
            self._document(folder_paths=["/vault/A", "/vault/B"]).compute_metadata_hash(),
        )


if __name__ == "__main__":
    unittest.main()
