from __future__ import annotations

from modssc.torch_backend_common import make_dtype_from_spec, make_resolve_device, make_to_tensor

from ..errors import OptionalDependencyError
from ..optional import import_torch

_torch = import_torch

resolve_device = make_resolve_device(
    torch_getter=lambda: _torch(),
    optional_dependency_error_cls=OptionalDependencyError,
    extra="inductive-torch",
)
dtype_from_spec = make_dtype_from_spec(torch_getter=lambda: _torch())
to_tensor = make_to_tensor(torch_getter=lambda: _torch())
