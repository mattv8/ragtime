"""Behavior contracts for the beta auto-merge serializer workflow."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "merge-serializer.yml"


def _pr(number: int, state: str = "behind", sha: str = "sha") -> dict[str, Any]:
    return {
        "number": number,
        "node_id": f"NODE{number}",
        "auto_merge": {"enabled_by": {"login": "someone"}},
        "draft": False,
        "mergeable_state": state,
        "head": {"sha": sha, "ref": f"feature-{number}", "repo": {"full_name": "org/repo"}},
        "base": {"ref": "beta"},
    }


class MergeSerializerTests(unittest.TestCase):
    def _script(self) -> str:
        workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        return workflow["jobs"]["advance-queue"]["steps"][0]["run"]

    def _run(self, fixture: dict[str, Any], token: bool = True) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
        fake_gh = textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, os, sys
            fixture = json.load(open(os.environ['GH_FIXTURE']))
            state_path = os.environ['GH_STATE']
            state = json.load(open(state_path)) if os.path.exists(state_path) else {}
            args = sys.argv[1:]
            with open(os.environ['GH_CALLS'], 'a') as calls:
                calls.write(json.dumps(args) + '\\n')
            if args[:2] == ['api', 'graphql']:
                if fixture.get('fail', {}).get('graphql'):
                    print('HTTP boundary failure', file=sys.stderr)
                    sys.exit(1)
                print(json.dumps(fixture.get('mutation', {'data': {'updatePullRequestBranch': {'pullRequest': {'number': 1}}}})))
                sys.exit(0)
            endpoint = next((arg for arg in args if arg.startswith('/')), '')
            route = endpoint.split('?', 1)[0]
            if route.endswith('/pulls'):
                value = fixture.get('list_pages', [])
            elif '/actions/runs' in route:
                sha = endpoint.split('head_sha=', 1)[1].split('&', 1)[0]
                value = fixture.get('runs', {}).get(sha, [{'workflow_runs': []}])
            elif '/pulls/' in route:
                number = route.rsplit('/', 1)[1]
                values = fixture.get('pulls', {}).get(number, [])
                index = state.get(number, 0)
                state[number] = index + 1
                json.dump(state, open(state_path, 'w'))
                value = values[min(index, len(values) - 1)] if values else {'message': 'missing'}
            else:
                value = {'message': 'unknown route'}
            if fixture.get('fail', {}).get(route):
                print('HTTP boundary failure', file=sys.stderr)
                sys.exit(1)
            print(json.dumps(value))
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "gh").write_text(fake_gh, encoding="utf-8")
            (path / "gh").chmod(0o755)
            (path / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
            calls = path / "calls.jsonl"
            environment = {
                **os.environ,
                "PATH": f"{path}:{os.environ['PATH']}",
                "GH_FIXTURE": str(path / "fixture.json"),
                "GH_STATE": str(path / "state.json"),
                "GH_CALLS": str(calls),
                "REPO": "org/repo",
            }
            environment.pop("GH_TOKEN", None)
            if token:
                environment["GH_TOKEN"] = "not-a-real-secret"
            result = subprocess.run(["bash", "-c", self._script()], text=True, capture_output=True, env=environment, check=False)
            recorded = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()] if calls.exists() else []
            return result, recorded

    def _assert_no_mutation(self, calls: list[list[str]]) -> None:
        self.assertFalse(any(call[:2] == ["api", "graphql"] for call in calls))

    def test_workflow_contract_and_missing_token_empty_queue(self) -> None:
        workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        self.assertEqual(workflow["on"]["push"]["branches"], ["beta"])
        self.assertEqual(workflow["on"]["schedule"][0]["cron"], "*/15 * * * *")
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertIn("github.ref == 'refs/heads/beta'", workflow["jobs"]["advance-queue"]["if"])
        self.assertNotIn("checkout", WORKFLOW.read_text(encoding="utf-8").lower())
        missing, calls = self._run({}, token=False)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("MERGE_SERIALIZER_TOKEN", missing.stderr)
        self.assertEqual(calls, [])
        empty, calls = self._run({"list_pages": [[]]})
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self._assert_no_mutation(calls)

    def test_filters_and_oldest_behind_selection(self) -> None:
        eligible = _pr(8)
        ignored = [_pr(1), _pr(2), _pr(3), _pr(4)]
        ignored[0]["auto_merge"] = None
        ignored[1]["draft"] = True
        ignored[2]["head"]["repo"]["full_name"] = "fork/repo"
        ignored[3]["head"]["ref"] = "main"
        result, calls = self._run({"list_pages": [ignored + [eligible, _pr(9)]], "pulls": {"8": [eligible, eligible], "9": [_pr(9), _pr(9)]}})
        self.assertEqual(result.returncode, 0, result.stderr)
        mutation = next(call for call in calls if call[:2] == ["api", "graphql"])
        self.assertIn("expectedHeadOid", " ".join(mutation))
        self.assertIn("REBASE", " ".join(mutation))
        self.assertIn("id=NODE8", mutation)
        self.assertIn("expectedHeadOid=sha", mutation)
        self.assertEqual(sum(call[:2] == ["api", "graphql"] for call in calls), 1)

    def test_deleted_fork_is_ignored_before_eligible_behind_pr(self) -> None:
        deleted_fork, eligible = _pr(1), _pr(2)
        deleted_fork["head"]["repo"] = None
        result, calls = self._run({"list_pages": [[deleted_fork, eligible]], "pulls": {"2": [eligible, eligible]}})
        self.assertEqual(result.returncode, 0, result.stderr)
        mutation = next(call for call in calls if call[:2] == ["api", "graphql"])
        self.assertIn("id=NODE2", mutation)

    def test_later_in_flight_pr_holds_older_behind_candidate(self) -> None:
        older, newer = _pr(1), _pr(2, "clean", "newsha")
        result, calls = self._run(
            {"list_pages": [[older, newer]], "pulls": {"1": [older], "2": [newer]}, "runs": {"newsha": [{"workflow_runs": [{"status": "in_progress"}]}]}}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("holding queue", result.stdout)
        self._assert_no_mutation(calls)

    def test_conflicts_and_completed_red_are_skipped(self) -> None:
        dirty, red, behind = _pr(1, "dirty"), _pr(2, "blocked", "redsha"), _pr(3)
        result, calls = self._run(
            {
                "list_pages": [[dirty, red, behind]],
                "pulls": {"1": [dirty], "2": [red], "3": [behind, behind]},
                "runs": {"redsha": [{"workflow_runs": [{"status": "completed"}]}]},
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(call[:2] == ["api", "graphql"] for call in calls), 1)

    def test_unknown_clean_and_paused_states_hold(self) -> None:
        for state, run in (("unknown", None), ("clean", {"status": "completed"}), ("blocked", {"status": "paused"})):
            with self.subTest(state=state, run=run):
                pr = _pr(1, state)
                runs = [{"workflow_runs": [] if run is None else [run]}]
                result, calls = self._run({"list_pages": [[pr]], "pulls": {"1": [pr]}, "runs": {"sha": runs}})
                self.assertEqual(result.returncode, 0, result.stderr)
                self._assert_no_mutation(calls)

    def test_pagination_api_errors_and_stale_candidate_do_not_advance(self) -> None:
        first, second = _pr(1), _pr(2)
        result, calls = self._run({"list_pages": [[first], [second]], "pulls": {"1": [first, first], "2": [second]}})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NODE1", " ".join(next(call for call in calls if call[:2] == ["api", "graphql"])))
        failed, _ = self._run({"list_pages": {"message": "denied"}})
        self.assertNotEqual(failed.returncode, 0)
        http_failed, calls = self._run({"fail": {"/repos/org/repo/pulls": True}})
        self.assertNotEqual(http_failed.returncode, 0)
        self._assert_no_mutation(calls)
        changed = _pr(1)
        changed_head = _pr(1, sha="new-sha")
        stale, calls = self._run({"list_pages": [[changed]], "pulls": {"1": [changed, changed_head]}})
        self.assertEqual(stale.returncode, 0, stale.stderr)
        self._assert_no_mutation(calls)

    def test_disabled_candidate_and_mutation_error_fail_without_retry(self) -> None:
        candidate, next_candidate = _pr(1), _pr(2)
        disabled = _pr(1)
        disabled["auto_merge"] = None
        stale, calls = self._run({"list_pages": [[candidate, next_candidate]], "pulls": {"1": [candidate, disabled], "2": [next_candidate]}})
        self.assertEqual(stale.returncode, 0, stale.stderr)
        self._assert_no_mutation(calls)
        failed, calls = self._run(
            {
                "list_pages": [[candidate, next_candidate]],
                "pulls": {"1": [candidate, candidate], "2": [next_candidate]},
                "mutation": {"errors": [{"message": "no permission"}]},
            }
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(sum(call[:2] == ["api", "graphql"] for call in calls), 1)


if __name__ == "__main__":
    unittest.main()
