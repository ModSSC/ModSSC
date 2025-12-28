from __future__ import annotations

import pytest

from modssc.data_augmentation.errors import OptionalDependencyError
from modssc.data_augmentation.optional import optional_import


def test_optional_import_success():
    math = optional_import("math", extra="math")
    assert math.sqrt(4) == 2.0


def test_optional_import_failure():
    with pytest.raises(OptionalDependencyError, match="Missing optional dependency"):
        optional_import("non_existent_module_xyz", extra="xyz")
