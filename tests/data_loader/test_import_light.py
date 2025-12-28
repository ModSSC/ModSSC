from __future__ import annotations

import subprocess
import sys
import textwrap


def test_import_is_light_in_subprocess() -> None:
    code = textwrap.dedent(
        """
        import sys
        import modssc.data_loader  # noqa: F401

        forbidden = [
            "torch",
            "tensorflow",
            "sklearn",
            "datasets",
            "torchvision",
            "torchaudio",
            "torch_geometric",
        ]
        imported = [name for name in forbidden if name in sys.modules]
        assert not imported, f"Heavy modules imported at import-time: {imported}"
        """
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
