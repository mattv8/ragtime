import json
import unittest
from unittest import mock

from langchain_core.tools import StructuredTool

from ragtime.rag.tool_skills import (
    ToolSkillBindingState,
    ToolSkillDefinition,
    build_tool_skill_control_tools,
    normalize_tool_skill_ids,
    resolve_tool_skill_bindings,
    search_tool_skill_catalog,
)


def _make_tool(name: str) -> StructuredTool:
    def _tool() -> str:
        return name

    return StructuredTool.from_function(_tool, name=name, description=f"Tool {name}")


def _make_tool_skill_definition(
    *,
    id: str,
    label: str,
    description: str,
    tool_names: list[str],
    tool_config_ids: list[str] | None = None,
    kind: str = "custom",
) -> ToolSkillDefinition:
    return ToolSkillDefinition(
        id=id,
        label=label,
        description=description,
        tool_names=tool_names,
        tool_config_ids=tool_config_ids or [],
        kind=kind,
    )


class ToolSkillHelpersTests(unittest.TestCase):
    def test_search_tool_skill_catalog_prefers_name_first_fuzzy_matches_for_ukg_query(self) -> None:
        catalog = [
            _make_tool_skill_definition(
                id="tool_config:ukg-id",
                label="UKG API",
                description="Accesses UKG employee, payroll, and cost center data.",
                tool_names=["request_ukg_api"],
                tool_config_ids=["ukg-id"],
                kind="http_api",
            ),
            _make_tool_skill_definition(
                id="tool_config:desc-only",
                label="Aardvark Connector",
                description="UKG API GET pay-info cost-centers cost-center-jobs exports for audits.",
                tool_names=["generic_http"],
                tool_config_ids=["desc-only"],
                kind="http_api",
            ),
        ]

        query = "UKG employees pay-info cost-centers cost-center-jobs API GET"
        results = search_tool_skill_catalog(catalog, query=query, limit=1)

        self.assertEqual(results, search_tool_skill_catalog(catalog, query=query, limit=1))
        self.assertEqual([item["id"] for item in results], ["tool_config:ukg-id"])
        self.assertEqual(results[0]["label"], "UKG API")
        self.assertEqual(results[0]["kind"], "http_api")
        self.assertEqual(results[0]["tool_names"], ["request_ukg_api"])
        self.assertNotIn("tool_config_ids", results[0])
        self.assertLessEqual(len(results), 1)

    def test_search_tool_skill_catalog_handles_ukg_typos_and_unrelated_queries(self) -> None:
        catalog = [
            _make_tool_skill_definition(
                id="tool_config:ukg-id",
                label="UKG API",
                description="Accesses UKG employee, payroll, and cost center data.",
                tool_names=["request_ukg_api"],
                tool_config_ids=["ukg-id"],
                kind="http_api",
            ),
            _make_tool_skill_definition(
                id="tool_config:pdm-id",
                label="SolidWorks PDM Database",
                description="Searches drawing and component metadata.",
                tool_names=["search_solidworks_pdm_database"],
                tool_config_ids=["pdm-id"],
                kind="database",
            ),
        ]

        typo_results = search_tool_skill_catalog(catalog, query="UKGG API", limit=8)
        self.assertEqual([item["id"] for item in typo_results], ["tool_config:ukg-id"])

        unrelated_results = search_tool_skill_catalog(catalog, query="lathe coolant spindle overload", limit=8)
        self.assertEqual(unrelated_results, [])

    def test_search_tool_skill_catalog_repeated_query_tokens_do_not_change_ordering(self) -> None:
        catalog = [
            _make_tool_skill_definition(
                id="tool_config:ukg-id",
                label="UKG API",
                description="Accesses UKG employee, payroll, and cost center data.",
                tool_names=["request_ukg_api"],
                tool_config_ids=["ukg-id"],
                kind="http_api",
            ),
            _make_tool_skill_definition(
                id="tool_config:weather-id",
                label="Weather Service",
                description="Gets current weather and forecast via API.",
                tool_names=["fetch_weather"],
                tool_config_ids=["weather-id"],
                kind="http_api",
            ),
        ]

        distinct_results = search_tool_skill_catalog(catalog, query="UKG API forecast", limit=8)
        repeated_results = search_tool_skill_catalog(catalog, query="UKG UKG UKG API API forecast forecast", limit=8)

        self.assertEqual([item["id"] for item in repeated_results], [item["id"] for item in distinct_results])

    def test_search_tool_skill_catalog_excludes_generic_http_api_weather_match_for_ukg_query(self) -> None:
        catalog = [
            _make_tool_skill_definition(
                id="tool_config:ukg-id",
                label="UKG API",
                description="Accesses UKG employee, payroll, and cost center data.",
                tool_names=["request_ukg_api"],
                tool_config_ids=["ukg-id"],
                kind="http_api",
            ),
            _make_tool_skill_definition(
                id="tool_config:weather-id",
                label="Weather Service",
                description="Gets current weather and forecast via API.",
                tool_names=["fetch_weather"],
                tool_config_ids=["weather-id"],
                kind="http_api",
            ),
        ]

        results = search_tool_skill_catalog(catalog, query="UKG employees pay-info cost-centers cost-center-jobs API GET", limit=8)

        self.assertEqual([item["id"] for item in results], ["tool_config:ukg-id"])

    def test_search_tool_skill_catalog_keeps_distinctive_description_searches(self) -> None:
        catalog = [
            _make_tool_skill_definition(
                id="tool_config:weather-id",
                label="Weather Service",
                description="Gets current weather and forecast via API.",
                tool_names=["fetch_weather"],
                tool_config_ids=["weather-id"],
                kind="http_api",
            ),
            _make_tool_skill_definition(
                id="tool_config:ukg-id",
                label="UKG API",
                description="Accesses UKG employee, payroll, and cost center data.",
                tool_names=["request_ukg_api"],
                tool_config_ids=["ukg-id"],
                kind="http_api",
            ),
        ]

        results = search_tool_skill_catalog(catalog, query="forecast", limit=8)

        self.assertEqual([item["id"] for item in results], ["tool_config:weather-id"])

    def test_normalize_tool_skill_ids_strips_empties_and_deduplicates(self) -> None:
        self.assertEqual(
            normalize_tool_skill_ids([" alpha ", "", None, "alpha", "beta", " beta "]),
            ["alpha", "beta"],
        )

    def test_normalize_tool_skill_ids_handles_bytes_and_bare_strings_safely(self) -> None:
        self.assertEqual(normalize_tool_skill_ids(" alpha "), ["alpha"])
        self.assertEqual(normalize_tool_skill_ids(b" beta "), ["beta"])

    def test_search_tool_skill_catalog_returns_empty_inventory_for_empty_catalog(self) -> None:
        self.assertEqual(search_tool_skill_catalog([], query="", limit=8), [])

    def test_search_tool_skill_catalog_is_deterministic_and_bounded(self) -> None:
        catalog = [
            ToolSkillDefinition(
                id="sql-read",
                label="SQL Reader",
                description="Reads customer records.",
                tool_names=["query_db"],
                tool_config_ids=["cfg-sql"],
                kind="database",
            ),
            ToolSkillDefinition(
                id="erp-read",
                label="ERP Search",
                description="Searches ERP records.",
                tool_names=["odoo_search"],
                tool_config_ids=["cfg-erp"],
                kind="odoo",
            ),
        ]

        empty_query = search_tool_skill_catalog(catalog, query="", limit=50)
        self.assertEqual([item["id"] for item in empty_query], ["erp-read", "sql-read"])
        self.assertEqual(len(empty_query), 2)
        self.assertLessEqual(len(empty_query), 20)

        query_results = search_tool_skill_catalog(catalog, query="database query_db", limit=0)
        self.assertEqual([item["id"] for item in query_results], ["sql-read"])
        self.assertEqual(query_results[0]["kind"], "database")
        self.assertNotIn("tool_config_ids", query_results[0])

    def test_search_tool_skill_catalog_clamps_limit_to_one_through_twenty(self) -> None:
        catalog = [
            ToolSkillDefinition(
                id=f"skill-{index:02d}",
                label=f"Skill {index:02d}",
                description=f"Description {index:02d}",
                tool_names=[f"tool_{index:02d}"],
                kind="custom",
            )
            for index in range(25)
        ]

        self.assertEqual(len(search_tool_skill_catalog(catalog, query="", limit=0)), 1)
        self.assertEqual(len(search_tool_skill_catalog(catalog, query="", limit=999)), 20)

    def test_search_tool_skill_catalog_clamps_description_length(self) -> None:
        catalog = [
            ToolSkillDefinition(
                id="long-desc",
                label="Long Description",
                description="x" * 700,
                tool_names=["tool_a"],
                kind="custom",
            )
        ]

        result = search_tool_skill_catalog(catalog, query="long", limit=1)[0]
        self.assertEqual(len(result["description"]), 500)

    def test_resolve_tool_skill_bindings_intersects_requested_ids_and_excludes_control_tools(self) -> None:
        eager_tool = _make_tool("always_on")
        optional_tools = {
            "query_db": _make_tool("query_db"),
            "odoo_search": _make_tool("odoo_search"),
        }
        catalog = [
            ToolSkillDefinition(
                id="sql-read",
                label="SQL Reader",
                description="Reads SQL data.",
                tool_names=["query_db"],
                tool_config_ids=["cfg-sql"],
                kind="database",
            ),
            ToolSkillDefinition(
                id="self-control",
                label="Control Tool",
                description="Should be excluded.",
                tool_names=["load_tool_skills"],
                tool_config_ids=["cfg-control"],
                kind="meta",
            ),
        ]

        resolution = resolve_tool_skill_bindings(
            eligible_catalog=catalog,
            requested_ids=["sql-read", "missing", "self-control"],
            eager_tools=[eager_tool],
            optional_tools_by_name=optional_tools,
        )

        self.assertEqual([tool.name for tool in resolution.eager_tools], ["always_on"])
        self.assertEqual([tool.name for tool in resolution.loaded_tools], ["query_db"])
        self.assertEqual(resolution.binding_state.requested_ids, ["missing", "self-control", "sql-read"])
        self.assertEqual(resolution.binding_state.effective_ids, ["sql-read"])
        self.assertEqual(resolution.hidden_optional_tool_config_ids, ["cfg-sql"])
        self.assertEqual([item.id for item in resolution.catalog], ["sql-read"])

    def test_resolve_tool_skill_bindings_preserves_ineligible_requested_ids_while_intersecting_effective_ids(self) -> None:
        resolution = resolve_tool_skill_bindings(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader",
                    description="Reads SQL data.",
                    tool_names=["query_db"],
                    kind="database",
                )
            ],
            requested_ids=[" retired-skill ", "sql-read", "retired-skill"],
            eager_tools=[],
            optional_tools_by_name={"query_db": _make_tool("query_db")},
        )

        self.assertEqual(resolution.binding_state.requested_ids, ["retired-skill", "sql-read"])
        self.assertEqual(resolution.binding_state.effective_ids, ["sql-read"])

    def test_resolve_tool_skill_bindings_dedupes_duplicate_catalog_ids(self) -> None:
        resolution = resolve_tool_skill_bindings(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader A",
                    description="Reads SQL data A.",
                    tool_names=["query_db"],
                    tool_config_ids=["cfg-a"],
                    kind="database",
                ),
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader B",
                    description="Reads SQL data B.",
                    tool_names=["query_db"],
                    tool_config_ids=["cfg-b"],
                    kind="database",
                ),
            ],
            requested_ids=["sql-read"],
            eager_tools=[],
            optional_tools_by_name={"query_db": _make_tool("query_db")},
        )

        self.assertEqual([item.id for item in resolution.catalog], ["sql-read"])
        self.assertEqual(resolution.hidden_optional_tool_config_ids, ["cfg-a"])
        self.assertEqual([tool.name for tool in resolution.loaded_tools], ["query_db"])

    def test_resolve_tool_skill_bindings_dedupes_loaded_tools_by_name_in_first_catalog_order(self) -> None:
        resolution = resolve_tool_skill_bindings(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="erp-read",
                    label="ERP Reader",
                    description="Reads ERP data.",
                    tool_names=["shared_query"],
                    kind="odoo",
                ),
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader",
                    description="Reads SQL data.",
                    tool_names=["shared_query"],
                    kind="database",
                ),
            ],
            requested_ids=["erp-read", "sql-read"],
            eager_tools=[],
            optional_tools_by_name={"shared_query": _make_tool("shared_query")},
        )

        self.assertEqual([tool.name for tool in resolution.loaded_tools], ["shared_query"])


class ToolSkillControlToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_tool_skills_adds_ids_persists_once_and_updates_binding_state(self) -> None:
        state = ToolSkillBindingState(requested_ids=["sql-read"], effective_ids=["sql-read"])
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader",
                    description="Reads SQL data.",
                    tool_names=["query_db"],
                    kind="database",
                ),
                ToolSkillDefinition(
                    id="erp-read",
                    label="ERP Search",
                    description="Searches ERP data.",
                    tool_names=["odoo_search"],
                    kind="odoo",
                ),
            ],
            binding_state=state,
            persist_requested_ids=persist,
        )

        load_tool = tools["load_tool_skills"]
        self.assertTrue(load_tool.return_direct)
        self.assertIn("standalone", load_tool.description.lower())

        result = json.loads(await load_tool.ainvoke({"ids": ["erp-read"]}))

        self.assertEqual(result["requested_ids"], ["erp-read", "sql-read"])
        self.assertEqual(result["effective_ids"], ["erp-read", "sql-read"])
        self.assertEqual(result["loaded_tool_names"], ["odoo_search"])
        self.assertTrue(result["bindings_changed"])
        self.assertEqual(result["transition_kind"], "load")
        self.assertEqual(state.requested_ids, ["erp-read", "sql-read"])
        self.assertEqual(state.effective_ids, ["erp-read", "sql-read"])
        self.assertTrue(state.bindings_changed)
        self.assertEqual(state.transition_kind, "load")
        persist.assert_awaited_once_with(["erp-read", "sql-read"])

    async def test_load_tool_skills_returns_deduplicated_loaded_tool_names_in_catalog_order(self) -> None:
        state = ToolSkillBindingState(requested_ids=[], effective_ids=[])
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="bundle-alpha",
                    label="Bundle Alpha",
                    description="Loads overlapping tools first.",
                    tool_names=["tool_b", "tool_a"],
                    kind="custom",
                ),
                ToolSkillDefinition(
                    id="bundle-beta",
                    label="Bundle Beta",
                    description="Loads overlapping tools second.",
                    tool_names=["tool_a", "tool_c"],
                    kind="custom",
                ),
            ],
            binding_state=state,
            persist_requested_ids=persist,
        )

        result = json.loads(await tools["load_tool_skills"].ainvoke({"ids": ["bundle-alpha", "bundle-beta"]}))

        self.assertEqual(result["requested_ids"], ["bundle-alpha", "bundle-beta"])
        self.assertEqual(result["effective_ids"], ["bundle-alpha", "bundle-beta"])
        self.assertEqual(result["loaded_tool_names"], ["tool_b", "tool_a", "tool_c"])
        persist.assert_awaited_once_with(["bundle-alpha", "bundle-beta"])

    async def test_load_tool_skills_excludes_tool_names_that_were_already_active(self) -> None:
        state = ToolSkillBindingState(requested_ids=["existing-skill"], effective_ids=["existing-skill"])
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="existing-skill",
                    label="Existing Skill",
                    description="Already active shared query.",
                    tool_names=["shared_query"],
                    kind="custom",
                ),
                ToolSkillDefinition(
                    id="new-skill",
                    label="New Skill",
                    description="Adds one new query plus overlap.",
                    tool_names=["shared_query", "new_query"],
                    kind="custom",
                ),
            ],
            binding_state=state,
            persist_requested_ids=persist,
        )

        result = json.loads(await tools["load_tool_skills"].ainvoke({"ids": ["new-skill"]}))

        self.assertEqual(result["requested_ids"], ["existing-skill", "new-skill"])
        self.assertEqual(result["effective_ids"], ["existing-skill", "new-skill"])
        self.assertEqual(result["loaded_tool_names"], ["new_query"])
        persist.assert_awaited_once_with(["existing-skill", "new-skill"])

    async def test_load_tool_skills_rejects_unknown_or_ineligible_ids_without_persisting(self) -> None:
        state = ToolSkillBindingState(requested_ids=["sql-read"], effective_ids=["sql-read"])
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader",
                    description="Reads SQL data.",
                    tool_names=["query_db"],
                    kind="database",
                )
            ],
            binding_state=state,
            persist_requested_ids=persist,
        )

        payload = json.loads(await tools["load_tool_skills"].ainvoke({"ids": ["missing"]}))
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], "unknown_tool_skill_ids")
        persist.assert_not_awaited()
        self.assertEqual(state.requested_ids, ["sql-read"])

    async def test_load_tool_skills_validation_error_uses_tool_level_output(self) -> None:
        state = ToolSkillBindingState(requested_ids=[], effective_ids=[])
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[],
            binding_state=state,
            persist_requested_ids=persist,
        )

        payload = json.loads(await tools["load_tool_skills"].ainvoke({}))

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], "invalid_tool_input")
        self.assertIn("ids", json.dumps(payload["details"]))
        persist.assert_not_awaited()

    async def test_load_tool_skills_skips_persist_for_noop(self) -> None:
        state = ToolSkillBindingState(requested_ids=["sql-read"], effective_ids=["sql-read"])
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader",
                    description="Reads SQL data.",
                    tool_names=["query_db"],
                    kind="database",
                )
            ],
            binding_state=state,
            persist_requested_ids=persist,
        )

        result = json.loads(await tools["load_tool_skills"].ainvoke({"ids": ["sql-read"]}))

        self.assertEqual(result["loaded_tool_names"], [])
        self.assertFalse(result["bindings_changed"])
        self.assertIsNone(result["transition_kind"])
        persist.assert_not_awaited()

    async def test_unload_tool_skills_removes_persisted_ineligible_ids(self) -> None:
        state = ToolSkillBindingState(
            requested_ids=["sql-read", "retired-skill"],
            effective_ids=["sql-read"],
        )
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader",
                    description="Reads SQL data.",
                    tool_names=["query_db"],
                    kind="database",
                )
            ],
            binding_state=state,
            persist_requested_ids=persist,
        )

        unload_tool = tools["unload_tool_skills"]
        self.assertTrue(unload_tool.return_direct)
        self.assertIn("standalone", unload_tool.description.lower())

        result = json.loads(await unload_tool.ainvoke({"ids": ["retired-skill"]}))

        self.assertEqual(result["requested_ids"], ["sql-read"])
        self.assertEqual(result["effective_ids"], ["sql-read"])
        self.assertTrue(result["bindings_changed"])
        self.assertEqual(result["transition_kind"], "unload")
        persist.assert_awaited_once_with(["sql-read"])

    async def test_unload_tool_skills_skips_persist_for_noop(self) -> None:
        state = ToolSkillBindingState(requested_ids=["sql-read"], effective_ids=["sql-read"])
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader",
                    description="Reads SQL data.",
                    tool_names=["query_db"],
                    kind="database",
                )
            ],
            binding_state=state,
            persist_requested_ids=persist,
        )

        result = json.loads(await tools["unload_tool_skills"].ainvoke({"ids": ["missing"]}))

        self.assertFalse(result["bindings_changed"])
        self.assertIsNone(result["transition_kind"])
        persist.assert_not_awaited()

    async def test_search_tool_skills_returns_untrusted_compact_json(self) -> None:
        state = ToolSkillBindingState(requested_ids=[], effective_ids=[])
        persist = mock.AsyncMock()
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="sql-read",
                    label="SQL Reader",
                    description="Reads SQL data.",
                    tool_names=["query_db"],
                    kind="database",
                )
            ],
            binding_state=state,
            persist_requested_ids=persist,
        )

        payload = json.loads(await tools["search_tool_skills"].ainvoke({"query": "sql", "limit": 8}))
        self.assertEqual(payload["status"], "ok")
        self.assertIn("untrusted metadata", payload["note"].lower())
        self.assertEqual(payload["results"][0]["id"], "sql-read")
