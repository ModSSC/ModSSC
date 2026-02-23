from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from importlib import import_module
from typing import Any, Literal

MethodStatus = Literal["implemented", "planned"]


def register_method_ref(
    *,
    registry: MutableMapping[str, Any],
    ref_factory: Callable[..., Any],
    method_id: str,
    import_path: str,
    status: MethodStatus = "implemented",
) -> None:
    if not method_id or not isinstance(method_id, str):
        raise ValueError("method_id must be a non-empty string")
    if ":" not in import_path:
        raise ValueError("import_path must be of the form 'pkg.module:ClassName'")
    existing = registry.get(method_id)
    if existing is not None and existing.import_path != import_path:
        raise ValueError(
            f"method_id {method_id!r} already registered with import_path={existing.import_path!r}"
        )
    if status not in {"implemented", "planned"}:
        raise ValueError("status must be 'implemented' or 'planned'")
    registry[method_id] = ref_factory(method_id=method_id, import_path=import_path, status=status)


def available_method_ids(*, registry: Mapping[str, Any], available_only: bool = True) -> list[str]:
    methods = sorted(registry.keys())
    if not available_only:
        return methods
    return [m for m in methods if registry[m].status != "planned"]


def load_method_class(
    *,
    registry: Mapping[str, Any],
    method_id: str,
    available_methods: Callable[[], list[str]],
) -> type[Any]:
    if method_id not in registry:
        raise KeyError(f"Unknown method_id: {method_id!r}. Available: {available_methods()}")
    ref = registry[method_id]
    mod_name, cls_name = ref.import_path.split(":")
    module = import_module(mod_name)
    return getattr(module, cls_name)


def make_register_method(*, registry: MutableMapping[str, Any], ref_factory: Callable[..., Any]):
    def register_method(
        method_id: str,
        import_path: str,
        *,
        status: MethodStatus = "implemented",
    ) -> None:
        register_method_ref(
            registry=registry,
            ref_factory=ref_factory,
            method_id=method_id,
            import_path=import_path,
            status=status,
        )

    return register_method


def make_available_methods(
    *,
    registry: Mapping[str, Any],
    ensure_builtins: Callable[[], None],
):
    def available_methods(*, available_only: bool = True) -> list[str]:
        ensure_builtins()
        return available_method_ids(registry=registry, available_only=available_only)

    return available_methods


def make_get_method_class(
    *,
    registry: Mapping[str, Any],
    ensure_builtins: Callable[[], None],
    available_methods: Callable[[], list[str]],
):
    def get_method_class(method_id: str) -> type[Any]:
        ensure_builtins()
        return load_method_class(
            registry=registry,
            method_id=method_id,
            available_methods=available_methods,
        )

    return get_method_class


def make_get_method_info(
    *,
    get_method_class: Callable[[str], type[Any]],
    method_info_cls: type[Any],
):
    def get_method_info(method_id: str):
        cls = get_method_class(method_id)
        info = getattr(cls, "info", None)
        if not isinstance(info, method_info_cls):
            raise TypeError(
                f"Method class {cls} must expose a class attribute `info: {method_info_cls.__name__}`"
            )
        return info

    return get_method_info


def make_debug_registry(
    *,
    registry: Mapping[str, Any],
    ensure_builtins: Callable[[], None],
):
    def _debug_registry() -> dict[str, Any]:
        ensure_builtins()
        return {k: v.import_path for k, v in registry.items()}

    return _debug_registry
