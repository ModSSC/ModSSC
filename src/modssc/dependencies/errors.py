from __future__ import annotations


class MissingExtraErrorMessageMixin:
    extra: str
    purpose: str | None

    def __str__(self) -> str:
        msg = f"Missing optional dependency extra: {self.extra!r}."
        if self.purpose:
            msg += f" Required for: {self.purpose}."
        msg += f' Install with: pip install "modssc[{self.extra}]"'
        return msg


class PackageOptionalDependencyMessageMixin:
    package: str
    extra: str
    message: str | None

    def __str__(self) -> str:
        base = self.message or f"Optional dependency {self.package!r} is required."
        return f'{base} Install with: pip install "modssc[{self.extra}]"'
