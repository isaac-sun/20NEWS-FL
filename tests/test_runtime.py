import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.export import export_results
from utils.runtime import require_cuda, resolve_gpu_profile


ROOT = Path(__file__).resolve().parents[1]


class GPUProfileTests(unittest.TestCase):
    def test_auto_profile_thresholds(self):
        self.assertEqual(resolve_gpu_profile("auto", 16.0).name, "t4")
        self.assertEqual(resolve_gpu_profile("auto", 24.0).name, "medium")
        self.assertEqual(resolve_gpu_profile("auto", 80.0).name, "large")

    def test_training_batch_is_invariant_across_profiles(self):
        profiles = [
            resolve_gpu_profile("t4", 80.0),
            resolve_gpu_profile("auto", 24.0),
            resolve_gpu_profile("large", 16.0),
        ]
        self.assertEqual([p.train_batch_size for p in profiles], [8, 8, 8])
        self.assertEqual([p.eval_batch_size for p in profiles], [16, 32, 64])

    def test_require_cuda_rejects_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
            require_cuda(True, "cpu")
        require_cuda(True, "cuda")
        require_cuda(False, "cpu")


class ArtifactTests(unittest.TestCase):
    def test_export_contains_effective_runtime_config(self):
        config = {
            "device": "cuda",
            "gpu_profile": "t4",
            "gpu_name": "Tesla T4",
            "gpu_memory_gb": 15.0,
            "batch_size": 8,
            "eval_batch_size": 16,
            "num_workers": 2,
        }
        with tempfile.TemporaryDirectory() as output_dir:
            path = export_results(
                [{"round": 0}],
                [{"final_global_loss": 1.0}],
                output_dir,
                experiment_config=config,
            )
            with pd.ExcelFile(path) as workbook:
                self.assertIn("experiment_config", workbook.sheet_names)
            exported = pd.read_excel(path, sheet_name="experiment_config")
            values = dict(zip(exported["parameter"], exported["value"]))
            self.assertEqual(values["gpu_profile"], "t4")
            self.assertEqual(values["eval_batch_size"], 16)

    def test_colab_notebook_preserves_torch_and_requires_cuda(self):
        notebook = json.loads((ROOT / "colab_demo.ipynb").read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("requirements-colab.txt", source)
        self.assertNotIn("pip install -q torch", source)
        self.assertIn("--require-cuda", source)
        self.assertIn("drive.mount", source)


if __name__ == "__main__":
    unittest.main()
