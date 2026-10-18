from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rwkv_hub import ensure_hf_checkpoint, push_adapter_snapshot


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
if __name__ == "__main__":
    unittest.main(verbosity=2)
