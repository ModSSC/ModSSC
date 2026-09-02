from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.utils import ensure_2d


@dataclass(frozen=True)
class TabularFeature:
    """A deterministic numeric or nominal feature description."""

    kind: str
    values: tuple[str, ...] = ()


def _is_missing(value: Any, missing_values: tuple[Any, ...]) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    for missing in missing_values:
        if missing is None:
            continue
        try:
            equal = value == missing
            if isinstance(equal, (bool, np.bool_)) and bool(equal):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _nominal_token(value: Any) -> str:
    return str(value).strip()


class TabularEncoder:
    """Encode mixed historical tabular data without third-party dependencies.

    Numeric values are represented directly and nominal values by stable integer
    codes. Missing values are represented by ``NaN`` in both cases. A declared
    schema fixes the nominal vocabulary, which makes train/test encoding and
    paper partition replays independent of the rows observed during fitting.
    """

    def __init__(
        self,
        *,
        feature_schema: Sequence[Mapping[str, Any] | str] | None,
        missing_values: Sequence[Any],
        classifier_name: str,
    ) -> None:
        self.feature_schema = feature_schema
        self.missing_values = tuple(missing_values)
        self.classifier_name = str(classifier_name)
        self.features_: tuple[TabularFeature, ...] | None = None

    def _declared_features(self, n_features: int) -> tuple[TabularFeature, ...] | None:
        schema = self.feature_schema
        if schema is None:
            return None
        if isinstance(schema, (str, bytes)) or len(schema) != int(n_features):
            raise SupervisedValidationError(
                f"{self.classifier_name} feature_schema must contain exactly "
                f"{int(n_features)} entries."
            )

        features: list[TabularFeature] = []
        for index, raw in enumerate(schema):
            if isinstance(raw, str):
                kind = raw
                values: Any = ()
            elif isinstance(raw, Mapping):
                kind = raw.get("type", raw.get("kind"))
                values = raw.get("values", ())
            else:
                raise SupervisedValidationError(
                    f"{self.classifier_name} feature_schema[{index}] must be a string or mapping."
                )
            normalized_kind = str(kind or "").strip().lower()
            if normalized_kind == "numeric":
                if values:
                    raise SupervisedValidationError(
                        f"{self.classifier_name} numeric feature_schema[{index}] cannot define "
                        "values."
                    )
                features.append(TabularFeature("numeric"))
                continue
            if normalized_kind != "nominal":
                raise SupervisedValidationError(
                    f"{self.classifier_name} feature_schema[{index}] type must be "
                    "'numeric' or 'nominal'."
                )
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise SupervisedValidationError(
                    f"{self.classifier_name} nominal feature_schema[{index}] requires a values "
                    "sequence."
                )
            nominal_values = tuple(_nominal_token(value) for value in values)
            if (
                not nominal_values
                or any(not value or value == "?" for value in nominal_values)
                or len(set(nominal_values)) != len(nominal_values)
            ):
                raise SupervisedValidationError(
                    f"{self.classifier_name} nominal feature_schema[{index}] has invalid values."
                )
            features.append(TabularFeature("nominal", nominal_values))
        return tuple(features)

    def _infer_features(self, array: np.ndarray) -> tuple[TabularFeature, ...]:
        features: list[TabularFeature] = []
        for column in range(int(array.shape[1])):
            known = [
                value
                for value in array[:, column].tolist()
                if not _is_missing(value, self.missing_values)
            ]
            numeric = True
            for value in known:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    numeric = False
                    break
                if not np.isfinite(number):
                    numeric = False
                    break
            if numeric:
                features.append(TabularFeature("numeric"))
                continue
            values = tuple(sorted({_nominal_token(value) for value in known}))
            features.append(TabularFeature("nominal", values))
        return tuple(features)

    def fit_transform(self, X: Any) -> np.ndarray:
        array = np.asarray(ensure_2d(X), dtype=object)
        if array.size == 0:
            raise SupervisedValidationError("X must be non-empty")
        features = self._declared_features(int(array.shape[1]))
        self.features_ = features if features is not None else self._infer_features(array)
        return self._transform_array(array)

    def transform(self, X: Any) -> np.ndarray:
        if self.features_ is None:
            raise RuntimeError("Tabular encoder is not fitted")
        array = np.asarray(ensure_2d(X), dtype=object)
        if int(array.shape[1]) != len(self.features_):
            raise SupervisedValidationError(
                f"{self.classifier_name} expected {len(self.features_)} features, got "
                f"{int(array.shape[1])}."
            )
        return self._transform_array(array)

    def _transform_array(self, array: np.ndarray) -> np.ndarray:
        if self.features_ is None:  # pragma: no cover - guarded by public methods
            raise RuntimeError("Tabular encoder is not fitted")
        encoded = np.full(array.shape, np.nan, dtype=np.float64)
        for column, feature in enumerate(self.features_):
            nominal_codes = {value: index for index, value in enumerate(feature.values)}
            for row, value in enumerate(array[:, column]):
                if _is_missing(value, self.missing_values):
                    continue
                if feature.kind == "numeric":
                    try:
                        number = float(value)
                    except (TypeError, ValueError) as exc:
                        raise SupervisedValidationError(
                            f"{self.classifier_name} feature {column} requires numeric values."
                        ) from exc
                    if not np.isfinite(number):
                        raise SupervisedValidationError(
                            f"{self.classifier_name} feature {column} requires finite numeric "
                            "values or a configured missing value."
                        )
                    encoded[row, column] = number
                    continue
                token = _nominal_token(value)
                if token not in nominal_codes:
                    raise SupervisedValidationError(
                        f"{self.classifier_name} feature {column} received nominal value "
                        f"{token!r}; expected one of {feature.values!r} or a configured missing "
                        "value."
                    )
                encoded[row, column] = float(nominal_codes[token])
        return encoded
