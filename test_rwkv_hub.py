from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rwkv_hub import (
    download_hub_checkpoint,
    ensure_hf_checkpoint,
    list_hub_checkpoint_versions,
    push_adapter_snapshot,
)


class EnsureHFCheckpointTests(unittest.TestCase):
    def test_local_mode_never_downloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            (model_dir / "release-manifest.json").write_text("{}")
            with patch("huggingface_hub.snapshot_download") as download:
                actual = ensure_hf_checkpoint(
                    "RWKV/test", "a" * 40, model_dir, fetch_from_hf=False
                )
            download.assert_not_called()
            self.assertEqual(actual, model_dir.resolve())

    def test_hf_mode_uses_exact_repo_revision_and_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            (model_dir / "release-manifest.json").write_text("{}")
            with patch("huggingface_hub.snapshot_download", return_value=str(model_dir)) as download:
                actual = ensure_hf_checkpoint(
                    "RWKV/test", "b" * 40, model_dir,
                    fetch_from_hf=True, token="secret-token",
                )
            self.assertEqual(actual, model_dir.resolve())
            download.assert_called_once_with(
                repo_id="RWKV/test",
                revision="b" * 40,
                local_dir=str(model_dir),
                cache_dir=None,
                token="secret-token",
            )

    def test_snapshot_without_release_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                ensure_hf_checkpoint(
                    "RWKV/test", "c" * 40, Path(temp_dir), fetch_from_hf=False
                )


class PushAdapterSnapshotTests(unittest.TestCase):
    def test_uploads_only_adapter_files_with_pinned_base_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            config_path = source / "adapter_config.json"
            weights_path = source / "adapter_model.safetensors"
            config_path.write_text(json.dumps({
                "base_model_name_or_path": "/local/model",
                "revision": None,
                "inference_mode": False,
            }))
            weights_path.write_bytes(b"safetensors-placeholder")

            from unittest.mock import MagicMock
            api = MagicMock()
            api.repo_info.return_value.private = True

            def inspect_upload(**kwargs):
                staging = Path(kwargs["folder_path"])
                self.assertEqual(
                    sorted(path.name for path in staging.iterdir()),
                    ["adapter_config.json", "adapter_metadata.json", "adapter_model.safetensors"],
                )
                config = json.loads((staging / "adapter_config.json").read_text())
                self.assertEqual(config["base_model_name_or_path"], "RWKV/test")
                self.assertEqual(config["revision"], "a" * 40)
                self.assertTrue(config["inference_mode"])
                metadata = json.loads((staging / "adapter_metadata.json").read_text())
                self.assertEqual(metadata["theta_hash"], "b" * 64)
                self.assertNotIn("secret-token", json.dumps(metadata))
                return type("Commit", (), {"commit_url": "https://huggingface.co/test/commit/abc"})()

            api.upload_folder.side_effect = inspect_upload
            metadata = {"theta_hash": "b" * 64, "theta_version": 1}
            with patch("huggingface_hub.HfApi", return_value=api) as hf_api:
                result = push_adapter_snapshot(
                    repo_id="test/adapter",
                    token="secret-token",
                    adapter_config_path=config_path,
                    adapter_weights_path=weights_path,
                    metadata=metadata,
                    path_in_repo="checkpoints/run/theta-000001",
                    base_model_id="RWKV/test",
                    base_model_revision="a" * 40,
                )

            hf_api.assert_called_once_with(token="secret-token")
            api.create_repo.assert_called_once_with(
                repo_id="test/adapter", repo_type="model", private=True,
                exist_ok=True, token="secret-token",
            )
            api.repo_info.assert_called_once_with(
                repo_id="test/adapter", repo_type="model", token="secret-token",
            )
            upload_kwargs = api.upload_folder.call_args.kwargs
            self.assertEqual(upload_kwargs["path_in_repo"], "checkpoints/run/theta-000001")
            self.assertEqual(upload_kwargs["token"], "secret-token")
            self.assertEqual(result, "https://huggingface.co/test/commit/abc")

    def test_refuses_missing_token(self):
        with self.assertRaisesRegex(ValueError, "write token"):
            push_adapter_snapshot(
                repo_id="test/adapter", token="", adapter_config_path=Path("missing"),
                adapter_weights_path=Path("missing"), metadata={}, path_in_repo="x",
                base_model_id="RWKV/test", base_model_revision="a" * 40,
            )

    def test_refuses_unsafe_hub_path(self):
        with self.assertRaisesRegex(ValueError, "relative path"):
            push_adapter_snapshot(
                repo_id="test/adapter", token="token", adapter_config_path=Path("missing"),
                adapter_weights_path=Path("missing"), metadata={}, path_in_repo="../escape",
                base_model_id="RWKV/test", base_model_revision="a" * 40,
            )

    def test_private_mode_refuses_an_existing_public_repository(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            config_path = source / "adapter_config.json"
            weights_path = source / "adapter_model.safetensors"
            config_path.write_text("{}")
            weights_path.write_bytes(b"weights")
            api = MagicMock()
            api.repo_info.return_value.private = False
            with patch("huggingface_hub.HfApi", return_value=api):
                with self.assertRaisesRegex(PermissionError, "is public"):
                    push_adapter_snapshot(
                        repo_id="test/public-adapter", token="token",
                        adapter_config_path=config_path, adapter_weights_path=weights_path,
                        metadata={}, path_in_repo="checkpoints/run/theta-000001",
                        base_model_id="RWKV/test", base_model_revision="a" * 40,
                    )
            api.upload_folder.assert_not_called()


class FakeHub:
    """A HfApi stand-in whose upload_folder copies the staged files into a local folder."""

    def __init__(self, root: Path, *, repo_private):
        self.root = root
        self.repo_private = repo_private
        self.uploads: list[dict] = []

    def api(self):
        from unittest.mock import MagicMock
        api = MagicMock()
        api.repo_info.return_value.private = self.repo_private

        def upload_folder(**kwargs):
            destination = self.root / kwargs["path_in_repo"]
            destination.mkdir(parents=True, exist_ok=True)
            for staged in Path(kwargs["folder_path"]).iterdir():
                if any(staged.match(pattern) for pattern in kwargs["allow_patterns"]):
                    (destination / staged.name).write_bytes(staged.read_bytes())
            self.uploads.append(kwargs)
            return type("Commit", (), {"commit_url": "https://huggingface.co/test/commit/abc"})()

        api.upload_folder.side_effect = upload_folder
        return api


class OptimizerUploadPolicyTests(unittest.TestCase):
    PATH = "checkpoints/run/theta-000002"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "adapter_config.json"
        self.weights = self.root / "adapter_model.safetensors"
        self.optimizer = self.root / "optimizer-src.safetensors"
        self.config.write_text("{}")
        self.weights.write_bytes(b"adapter-weights")
        self.optimizer.write_bytes(b"optimizer-moments")

    def push(self, *, repo_private, policy, private=True, optimizer="present") -> tuple[FakeHub, dict, dict]:
        hub = FakeHub(self.root / "hub", repo_private=repo_private)
        report: dict = {}
        metadata = {"theta_hash": "b" * 64}
        with patch("huggingface_hub.HfApi", return_value=hub.api()):
            push_adapter_snapshot(
                repo_id="test/adapter", token="token", adapter_config_path=self.config,
                adapter_weights_path=self.weights, metadata=metadata, path_in_repo=self.PATH,
                base_model_id="RWKV/test", base_model_revision="a" * 40, private=private,
                optimizer_weights_path=self.optimizer if optimizer == "present" else None,
                optimizer_policy=policy, report=report,
            )
        self.assertEqual(metadata, {"theta_hash": "b" * 64}, "caller's metadata must not be modified")
        return hub, report, json.loads((hub.root / self.PATH / "adapter_metadata.json").read_text())

    def uploaded_names(self, hub: FakeHub) -> list[str]:
        return sorted(path.name for path in (hub.root / self.PATH).iterdir())

    def test_auto_uploads_optimizer_to_a_private_repository(self):
        hub, report, metadata = self.push(repo_private=True, policy="auto")
        self.assertIn("optimizer.safetensors", self.uploaded_names(hub))
        self.assertEqual((hub.root / self.PATH / "optimizer.safetensors").read_bytes(), b"optimizer-moments")
        self.assertTrue(metadata["optimizer_included"])
        import hashlib
        self.assertEqual(metadata["optimizer_sha256"], hashlib.sha256(b"optimizer-moments").hexdigest())
        self.assertEqual(report, {"repo_private": True, "optimizer_uploaded": True})

    def test_auto_withholds_optimizer_from_a_public_repository(self):
        hub, report, metadata = self.push(repo_private=False, policy="auto", private=False)
        self.assertEqual(self.uploaded_names(hub),
                         ["adapter_config.json", "adapter_metadata.json", "adapter_model.safetensors"])
        self.assertFalse(metadata["optimizer_included"])
        self.assertNotIn("optimizer_sha256", metadata)
        self.assertEqual(report, {"repo_private": False, "optimizer_uploaded": False})

    def test_auto_withholds_optimizer_when_visibility_is_unknown(self):
        hub, report, _ = self.push(repo_private=None, policy="auto", private=False)
        self.assertNotIn("optimizer.safetensors", self.uploaded_names(hub))
        self.assertFalse(report["optimizer_uploaded"])

    def test_auto_follows_the_repository_not_the_requested_visibility(self):
        # --hf-adapter-public does not make a pre-existing private repo public.
        hub, report, _ = self.push(repo_private=True, policy="auto", private=False)
        self.assertIn("optimizer.safetensors", self.uploaded_names(hub))
        self.assertTrue(report["optimizer_uploaded"])

    def test_always_uploads_to_a_public_repository_when_explicitly_chosen(self):
        hub, report, metadata = self.push(repo_private=False, policy="always", private=False)
        self.assertIn("optimizer.safetensors", self.uploaded_names(hub))
        self.assertTrue(metadata["optimizer_included"])
        self.assertTrue(report["optimizer_uploaded"])

    def test_never_withholds_optimizer_even_from_a_private_repository(self):
        hub, report, metadata = self.push(repo_private=True, policy="never")
        self.assertNotIn("optimizer.safetensors", self.uploaded_names(hub))
        self.assertFalse(metadata["optimizer_included"])
        self.assertFalse(report["optimizer_uploaded"])

    def test_nothing_to_upload_without_an_optimizer_file(self):
        hub, report, metadata = self.push(repo_private=True, policy="always", optimizer="absent")
        self.assertNotIn("optimizer.safetensors", self.uploaded_names(hub))
        self.assertFalse(metadata["optimizer_included"])

    def test_unknown_policy_is_rejected_before_any_network_call(self):
        with patch("huggingface_hub.HfApi") as api:
            with self.assertRaisesRegex(ValueError, "optimizer_policy"):
                push_adapter_snapshot(
                    repo_id="test/adapter", token="token", adapter_config_path=self.config,
                    adapter_weights_path=self.weights, metadata={}, path_in_repo=self.PATH,
                    base_model_id="RWKV/test", base_model_revision="a" * 40,
                    optimizer_weights_path=self.optimizer, optimizer_policy="sometimes",
                )
        api.assert_not_called()

    def test_private_mode_still_refuses_a_public_repo_regardless_of_policy(self):
        with self.assertRaisesRegex(PermissionError, "is public"):
            self.push(repo_private=False, policy="always", private=True)


class HubCheckpointDiscoveryTests(unittest.TestCase):
    def test_lists_only_complete_versions_of_the_requested_run_newest_first(self):
        from unittest.mock import MagicMock
        api = MagicMock()
        api.list_repo_files.return_value = [
            "README.md",
            "checkpoints/run-a/theta-000002/adapter_model.safetensors",
            "checkpoints/run-a/theta-000002/adapter_metadata.json",
            "checkpoints/run-a/theta-000010/adapter_model.safetensors",
            "checkpoints/run-a/theta-000010/adapter_metadata.json",
            "checkpoints/run-a/theta-000010/optimizer.safetensors",
            "checkpoints/run-a/theta-000012/adapter_model.safetensors",  # metadata missing
            "checkpoints/run-b/theta-000030/adapter_model.safetensors",
            "checkpoints/run-b/theta-000030/adapter_metadata.json",
        ]
        with patch("huggingface_hub.HfApi", return_value=api):
            versions = list_hub_checkpoint_versions(repo_id="test/adapter", run_id="run-a", token="t")
        self.assertEqual(versions, [10, 2])

    def test_unsafe_run_ids_are_rejected(self):
        for run_id in ("", "../other", "a/b", "run*", "-leading"):
            with self.subTest(run_id=run_id):
                with self.assertRaisesRegex(ValueError, "unsafe run id"):
                    list_hub_checkpoint_versions(repo_id="test/adapter", run_id=run_id, token=None)
                with self.assertRaisesRegex(ValueError, "unsafe run id"):
                    download_hub_checkpoint(repo_id="test/adapter", run_id=run_id, version=1,
                                            token=None, local_dir=Path("unused"))

    def test_download_asks_for_one_version_folder_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "dl"

            def fake_download(**kwargs):
                folder = Path(kwargs["local_dir"]) / "checkpoints/run-a/theta-000007"
                folder.mkdir(parents=True)
                return kwargs["local_dir"]

            with patch("huggingface_hub.snapshot_download", side_effect=fake_download) as download:
                folder = download_hub_checkpoint(repo_id="test/adapter", run_id="run-a", version=7,
                                                 token="t", local_dir=destination)
            self.assertEqual(folder, destination / "checkpoints/run-a/theta-000007")
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs["allow_patterns"], ["checkpoints/run-a/theta-000007/*"])
            self.assertEqual((kwargs["repo_id"], kwargs["repo_type"], kwargs["token"]),
                             ("test/adapter", "model", "t"))

    def test_download_that_produces_nothing_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("huggingface_hub.snapshot_download"):
                with self.assertRaises(FileNotFoundError):
                    download_hub_checkpoint(repo_id="test/adapter", run_id="run-a", version=7,
                                            token=None, local_dir=Path(temp_dir))


if __name__ == "__main__":
    unittest.main(verbosity=2)
