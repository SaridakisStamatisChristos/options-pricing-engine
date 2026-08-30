"""Guard the exact-commit release dependency and supply-chain architecture."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _load(name: str) -> dict[str, Any]:
    # BaseLoader follows YAML 1.2-like scalar behavior for GitHub's special
    # ``on`` key and is sufficient for structural workflow assertions.
    loaded = yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def _run_commands(job: dict[str, Any]) -> str:
    return "\n".join(
        str(step.get("run", "")) for step in job.get("steps", []) if isinstance(step, dict)
    )


def _external_action_uses(workflow: dict[str, Any]) -> list[str]:
    uses: list[str] = []
    for job in workflow.get("jobs", {}).values():
        if not isinstance(job, dict):
            continue
        job_uses = job.get("uses")
        if isinstance(job_uses, str) and not job_uses.startswith("./"):
            uses.append(job_uses)
        for step in job.get("steps", []):
            if not isinstance(step, dict):
                continue
            step_uses = step.get("uses")
            if isinstance(step_uses, str) and not step_uses.startswith("./"):
                uses.append(step_uses)
    return uses


def test_release_is_callable_only_and_contains_no_duplicate_quality_suite() -> None:
    release = _load("release.yml")

    assert set(release["on"]) == {"workflow_call"}
    assert release["permissions"] == {}
    publish = release["jobs"]["publish"]
    assert publish["permissions"] == {
        "contents": "write",
        "id-token": "write",
        "attestations": "write",
    }

    commands = _run_commands(publish)
    for duplicate_gate in (
        "pytest",
        "ruff check",
        "ruff format",
        "mypy",
        "bandit",
        "pip-audit",
        "trivy",
        "uv build",
        "twine check",
    ):
        assert duplicate_gate not in commands


def test_release_waits_for_every_required_gate() -> None:
    ci = _load("ci.yml")
    release = ci["jobs"]["release"]

    assert set(release["needs"]) == {
        "tests",
        "minimum-versions",
        "quality",
        "container",
        "codeql",
    }
    assert release["uses"] == "./.github/workflows/release.yml"
    assert "github.event_name == 'push'" in release["if"]
    assert "github.ref == 'refs/heads/main'" in release["if"]
    assert release["with"] == {
        "source_sha": "${{ github.sha }}",
        "artifact_name": "distributions-${{ github.sha }}",
    }

    quality = _run_commands(ci["jobs"]["quality"])
    for required_gate in (
        "actionlint",
        "ruff format --check",
        "ruff check",
        "mypy src",
        "bandit -q -r src -ll",
        "--cov=options_engine",
        "uv build",
        "twine check",
        "pip-audit --strict",
        "sha256sum --check SHA256SUMS",
    ):
        assert required_gate in quality

    minimum = _run_commands(ci["jobs"]["minimum-versions"])
    assert "--resolution lowest-direct" in minimum
    assert "pytest" in minimum
    assert "pip-audit" in minimum
    assert ci["jobs"]["codeql"]["uses"] == "./.github/workflows/codeql.yml"
    assert any(
        "aquasecurity/trivy-action@" in step.get("uses", "")
        for step in ci["jobs"]["container"]["steps"]
    )


def test_codeql_has_no_parallel_push_or_pull_request_trigger() -> None:
    codeql = _load("codeql.yml")

    assert set(codeql["on"]) == {"workflow_call", "workflow_dispatch", "schedule"}
    assert "push" not in codeql["on"]
    assert "pull_request" not in codeql["on"]


def test_release_reuses_tested_artifacts_and_attests_distributions() -> None:
    release = _load("release.yml")
    publish = release["jobs"]["publish"]
    steps = publish["steps"]
    commands = _run_commands(publish)

    download = next(step for step in steps if "actions/download-artifact@" in step.get("uses", ""))
    attestation = next(
        step for step in steps if "actions/attest-build-provenance@" in step.get("uses", "")
    )
    assert download["with"]["name"] == "${{ inputs.artifact_name }}"
    assert "dist/*.whl" in attestation["with"]["subject-path"]
    assert "dist/*.tar.gz" in attestation["with"]["subject-path"]
    assert '"${SOURCE_SHA}" != "${GITHUB_SHA}"' in commands
    assert "sha256sum --check SHA256SUMS" in commands
    assert "tag_commit" in commands
    assert '--target "${SOURCE_SHA}"' in commands


def test_all_external_actions_are_immutably_pinned() -> None:
    for workflow_path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = _load(workflow_path.name)
        for uses in _external_action_uses(workflow):
            action, separator, ref = uses.rpartition("@")
            assert action and separator, f"missing action ref in {workflow_path}: {uses}"
            assert FULL_SHA.fullmatch(ref), (
                f"external action is not pinned to a full commit SHA in {workflow_path}: {uses}"
            )

        for job in workflow.get("jobs", {}).values():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps", []):
                if not isinstance(step, dict):
                    continue
                if str(step.get("uses", "")).startswith("actions/checkout@"):
                    assert step.get("with", {}).get("persist-credentials") == "false"
