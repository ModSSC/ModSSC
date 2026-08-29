from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional torch dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import modssc.inductive.methods.simclr_v2 as simclr_v2
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.simclr_v2 import SimCLRv2Method, SimCLRv2Spec
from modssc.inductive.types import DeviceSpec, InductiveDataset
from modssc.runtime.contracts import ModelContract

from ..conftest import (
    SimpleNet,
    make_model_bundle,
    make_numpy_dataset,
    make_torch_dataset,
    make_torch_ssl_dataset,
)


class _MetaNet(torch.nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.feat = torch.nn.Linear(2, 2, bias=False)
        self.proj = torch.nn.Linear(2, 2, bias=False)
        self.fc = torch.nn.Linear(2, n_classes, bias=False)

    def _unwrap(self, x):
        return x["x"] if isinstance(x, dict) else x

    def encode(self, x):
        return self.feat(self._unwrap(x))

    def forward_projection(self, x):
        return self.proj(self.encode(x))

    def project(self, feat):
        return self.proj(feat)

    def forward_logits(self, x):
        feat = self.encode(x)
        return {"logits": self.fc(feat)}

    def forward_classifier(self, x):
        return {"logits": self.fc(self._unwrap(x))}

    def classify(self, x):
        return {"logits": self.fc(self._unwrap(x))}

    def forward_head(self, x):
        return {"logits": self.fc(self._unwrap(x))}

    def forward(self, x):
        feat = self.encode(x)
        proj = self.proj(feat)
        logits = self.fc(feat)
        return {"feat": feat, "proj": proj, "logits": logits}


class _CountingMetaNet(_MetaNet):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__(n_classes=n_classes)
        self.batch_sizes: list[int] = []

    def forward(self, x):
        x_in = x["x"] if isinstance(x, dict) else x
        self.batch_sizes.append(int(x_in.shape[0]))
        return super().forward(x)


class _BadLogits1D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x["x"] if isinstance(x, dict) else x
        return torch.zeros((int(x.shape[0]),), device=x.device)


def _make_bundle(
    model: torch.nn.Module | None = None,
    *,
    meta: dict | None = None,
    contract: ModelContract | None = None,
) -> TorchModelBundle:
    bundle_model = model if model is not None else SimpleNet()
    optimizer = torch.optim.SGD(bundle_model.parameters(), lr=0.1)
    return TorchModelBundle(
        model=bundle_model,
        optimizer=optimizer,
        meta=meta,
        contract=contract,
    )


def _make_projection_bundle() -> TorchModelBundle:
    model = _MetaNet()
    return _make_bundle(
        model,
        meta={
            "forward_features": model.encode,
            "forward_projection": model.forward_projection,
            "forward_head": model.forward_head,
        },
        contract=ModelContract(
            outputs=frozenset({"feat", "forward_features", "forward_projection", "logits", "proj"}),
            input_representations=frozenset({"dense"}),
            input_dtype_kinds=frozenset({"float"}),
            input_ranks=frozenset({2}),
            source="test.simclrv2_projection",
        ),
    )


def _make_spec(**overrides) -> SimCLRv2Spec:
    base = SimCLRv2Spec(
        pretrain_bundle=_make_projection_bundle(),
        finetune_bundle=make_model_bundle(),
        student_bundle=None,
        temperature=0.5,
        distill_temperature=1.0,
        alpha=0.5,
        batch_size=2,
        pretrain_epochs=1,
        finetune_epochs=1,
        distill_epochs=1,
        transfer_pretrain=True,
        use_labeled_in_distill=True,
        freeze_bn=False,
        detach_target=True,
    )
    return SimCLRv2Spec(**{**base.__dict__, **overrides})


def _patch_single_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simclr_v2, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(simclr_v2, "num_batches", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        simclr_v2, "cycle_batch_indices", lambda *_args, **_kwargs: iter([torch.tensor([0, 1])])
    )
    monkeypatch.setattr(
        simclr_v2,
        "cycle_batches",
        lambda X, y, **_kwargs: iter([(X[:2], y[:2])]),
    )


def _iter_outputs(*outputs) -> Iterator[torch.Tensor]:
    return iter(outputs)


def test_simclr_v2_tensor_helpers() -> None:
    tensor = torch.randn(2, 2)
    assert simclr_v2._as_tensor(tensor, name="x") is tensor
    assert simclr_v2._tensor_from_output(tensor, keys=("feat",), name="out") is tensor
    assert simclr_v2._tensor_from_output({"feat": tensor}, keys=("feat",), name="out") is tensor
    assert simclr_v2._tensor_from_output((tensor, "ignored"), keys=("feat",), name="out") is tensor

    with pytest.raises(InductiveValidationError, match="x must be a torch.Tensor"):
        simclr_v2._as_tensor("bad", name="x")

    with pytest.raises(InductiveValidationError, match="out\\[feat\\] must be a torch.Tensor"):
        simclr_v2._tensor_from_output({"feat": "bad"}, keys=("feat",), name="out")

    with pytest.raises(InductiveValidationError, match="mapping with keys"):
        simclr_v2._tensor_from_output({"other": tensor}, keys=("feat",), name="out")


def test_simclr_v2_contract_requires_a_real_projection_for_pretraining() -> None:
    spec = SimCLRv2Spec(
        pretrain_epochs=1,
        finetune_epochs=0,
        distill_epochs=0,
    )
    contract = SimCLRv2Method.execution_contract(
        spec,
        SimCLRv2Method.info.capabilities,
        SimCLRv2Method.info.model_binding,
    )
    requirement = contract.components[0]

    assert frozenset({"projection"}) in requirement.output_alternatives
    assert frozenset({"forward_projection"}) in requirement.output_alternatives
    assert frozenset({"feat"}) not in requirement.output_alternatives
    assert frozenset({"logits"}) not in requirement.output_alternatives


def test_simclr_v2_forward_helpers_cover_meta_paths() -> None:
    x = torch.randn(3, 2)
    model = _MetaNet()

    assert simclr_v2._forward_features(model, {"forward_features": model.encode}, x).shape == (3, 2)
    assert simclr_v2._forward_features(model, {"feature_extractor": model.encode}, x).shape == (
        3,
        2,
    )
    assert simclr_v2._forward_features(model, {"encoder": model.encode}, x).shape == (3, 2)
    assert simclr_v2._forward_features(model, {"forward_features": 1}, x).shape == (3, 2)
    assert simclr_v2._forward_features(model, None, x).shape == (3, 2)
    for feature_key in ("features", "embedding"):

        def feature_model(_value, key=feature_key):
            return {key: torch.ones(3, 2)}

        assert simclr_v2._forward_features(feature_model, None, x).shape == (3, 2)
    with pytest.raises(InductiveValidationError, match="requires an explicit encoder feature"):
        simclr_v2._forward_features(lambda _value: {"logits": torch.ones(3, 2)}, None, x)

    assert simclr_v2._forward_projection(
        model, {"forward_projection": model.forward_projection}, x
    ).shape == (3, 2)
    assert simclr_v2._forward_projection(
        model,
        {"projection_head": model.project, "forward_features": model.encode},
        x,
    ).shape == (3, 2)
    assert simclr_v2._forward_projection(
        model,
        {"projector": model.project, "encoder": model.encode},
        x,
    ).shape == (3, 2)
    assert simclr_v2._forward_projection(
        model, {"forward_projection": 1, "projector": 1}, x
    ).shape == (
        3,
        2,
    )
    assert simclr_v2._forward_projection(model, None, x).shape == (3, 2)
    for projection_key in ("projection", "z"):

        def projection_model(_value, key=projection_key):
            return {key: torch.ones(3, 2)}

        assert simclr_v2._forward_projection(projection_model, None, x).shape == (3, 2)

    with pytest.raises(InductiveValidationError, match="requires an explicit projection head"):
        simclr_v2._forward_projection(
            lambda _value: {"logits": torch.ones(3, 2)},
            None,
            x,
        )

    def declared_tensor_model(_value):
        return torch.ones(3, 2)

    assert simclr_v2._forward_projection(
        declared_tensor_model,
        None,
        x,
        contract=ModelContract(
            outputs=frozenset({"projection"}),
            source="test.declared_tensor_projection",
        ),
    ).shape == (3, 2)

    logits_only = _BadLogits1D()
    with pytest.raises(InductiveValidationError, match="requires an explicit encoder feature"):
        simclr_v2._forward_features(logits_only, None, x)
    with pytest.raises(InductiveValidationError, match="requires an explicit projection head"):
        simclr_v2._forward_projection(logits_only, None, x)
    with pytest.raises(InductiveValidationError, match="requires a declared encoder feature"):
        simclr_v2._forward_projection(
            model,
            {"projection_head": model.project},
            x,
            contract=ModelContract(
                outputs=frozenset({"logits", "projection_head"}),
                source="test.contradictory_projection",
            ),
        )

    assert simclr_v2._forward_logits(model, {"forward_logits": model.forward_logits}, x).shape == (
        3,
        2,
    )
    assert simclr_v2._forward_logits(
        model,
        {"forward_classifier": model.forward_classifier},
        x,
    ).shape == (3, 2)
    assert simclr_v2._forward_logits(model, {"classifier": model.classify}, x).shape == (3, 2)
    assert simclr_v2._forward_logits(
        model,
        {"forward_head": model.forward_head, "forward_features": model.encode},
        x,
    ).shape == (3, 2)
    assert simclr_v2._forward_logits(
        model,
        {"head": model.classify, "encoder": model.encode},
        x,
    ).shape == (3, 2)
    assert simclr_v2._forward_logits(model, {"forward_logits": 1, "head": 1}, x).shape == (3, 2)
    assert simclr_v2._forward_logits(model, {"head": model.classify}, x).shape == (3, 2)
    assert simclr_v2._forward_logits(model, None, x).shape == (3, 2)


def test_simclr_v2_rebind_meta_and_check_distill_models() -> None:
    source = _MetaNet()
    target = _MetaNet()
    shared = torch.nn.Linear(2, 2, bias=False)

    meta = {
        "forward_features": source.encode,
        "projector": source.project,
        "callable": lambda x: x,
        "value": 3,
    }
    rebound = simclr_v2._rebind_meta(meta, source=source, target=target)
    assert rebound is not None
    assert rebound["forward_features"].__self__ is target
    assert rebound["projector"].__self__ is target
    assert rebound["callable"] is meta["callable"]
    assert rebound["value"] == 3
    assert simclr_v2._rebind_meta(None, source=source, target=target) is None
    assert simclr_v2._rebind_meta(123, source=source, target=target) == 123

    shadow_target = type("ShadowTarget", (), {"encode": 1})()
    rebound_shadow = simclr_v2._rebind_meta(
        {"forward_features": source.encode}, source=source, target=shadow_target
    )
    assert rebound_shadow["forward_features"].__self__ is source

    simclr_v2._check_distill_models(_MetaNet(), _MetaNet())

    with pytest.raises(InductiveValidationError, match="must be distinct"):
        simclr_v2._check_distill_models(source, source)

    with pytest.raises(InductiveValidationError, match="must not share parameters"):
        simclr_v2._check_distill_models(torch.nn.Sequential(shared), torch.nn.Sequential(shared))


def test_simclr_v2_projection_state_must_be_model_owned_and_optimized() -> None:
    valid = _make_projection_bundle()
    simclr_v2._validate_projection_optimization(valid)
    assert any(key.startswith("proj.") for key in valid.model.state_dict())

    restored = _MetaNet()
    restored.load_state_dict(valid.model.state_dict())
    assert torch.allclose(restored.proj.weight, valid.model.proj.weight)

    model = _MetaNet()
    incomplete = TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(
            [*model.feat.parameters(), *model.fc.parameters()],
            lr=0.1,
        ),
        meta={"forward_projection": model.forward_projection},
    )
    with pytest.raises(InductiveValidationError, match="must own every trainable"):
        simclr_v2._validate_projection_optimization(incomplete)

    base = SimpleNet()
    external = torch.nn.Linear(2, 2)
    external_bundle = TorchModelBundle(
        model=base,
        optimizer=torch.optim.SGD(
            [*base.parameters(), *external.parameters()],
            lr=0.1,
        ),
        meta={"projection_head": external},
    )
    with pytest.raises(InductiveValidationError, match="registered inside bundle.model"):
        simclr_v2._validate_projection_optimization(external_bundle)

    functional = _make_bundle(_MetaNet(), meta={"forward_projection": lambda value: value})
    with pytest.raises(InductiveValidationError, match="bound module method"):
        simclr_v2._validate_projection_optimization(functional)

    no_meta = _make_bundle(_MetaNet())
    simclr_v2._validate_projection_optimization(no_meta)
    no_projection = _make_bundle(_MetaNet(), meta={"forward_features": 1})
    simclr_v2._validate_projection_optimization(no_projection)

    projector_model = _MetaNet()
    projector_bundle = _make_bundle(
        projector_model,
        meta={"projector": projector_model.proj},
    )
    simclr_v2._validate_projection_optimization(projector_bundle)

    class _ParameterlessProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.projector = torch.nn.Identity()

        def forward(self, value):
            return self.projector(self.encoder(value))

    parameterless_model = _ParameterlessProjection()
    parameterless_bundle = _make_bundle(
        parameterless_model,
        meta={"projection_head": parameterless_model.projector},
    )
    with pytest.raises(InductiveValidationError, match="must expose trainable parameters"):
        simclr_v2._validate_projection_optimization(parameterless_bundle)


def test_simclr_v2_contract_fallback_and_student_labeled_input_branches() -> None:
    spec = SimCLRv2Spec(
        pretrain_epochs=0,
        finetune_epochs=0,
        distill_epochs=0,
    )
    fallback = SimCLRv2Method.execution_contract(
        spec,
        SimCLRv2Method.info.capabilities,
        None,
    )
    assert fallback.components == ()

    with_student = SimCLRv2Spec(
        pretrain_epochs=0,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=True,
        student_bundle=_make_bundle(),
    )
    contract = SimCLRv2Method.execution_contract(
        with_student,
        SimCLRv2Method.info.capabilities,
        SimCLRv2Method.info.model_binding,
    )
    student = next(
        component for component in contract.components if component.slot == "student_bundle"
    )
    assert "fit.X_l" in student.input_roles

    shared_slot = SimCLRv2Spec(
        pretrain_epochs=1,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=True,
    )
    shared_contract = SimCLRv2Method.execution_contract(
        shared_slot,
        SimCLRv2Method.info.capabilities,
        SimCLRv2Method.info.model_binding,
    )
    assert len(shared_contract.components) == 1
    assert "fit.X_l" in shared_contract.components[0].input_roles

    selected_finetune = SimCLRv2Spec(
        pretrain_bundle=None,
        finetune_bundle=_make_projection_bundle(),
        pretrain_epochs=1,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=True,
    )
    selected_contract = SimCLRv2Method.execution_contract(
        selected_finetune,
        SimCLRv2Method.info.capabilities,
        SimCLRv2Method.info.model_binding,
    )
    assert len(selected_contract.components) == 1
    assert "fit.X_l" in selected_contract.components[0].input_roles

    finetune_only = SimCLRv2Method.execution_contract(
        SimCLRv2Spec(pretrain_epochs=0, finetune_epochs=1, distill_epochs=0),
        SimCLRv2Method.info.capabilities,
        SimCLRv2Method.info.model_binding,
    )
    assert finetune_only.components[0].input_roles == ("fit.X_l",)

    distill_only = SimCLRv2Method.execution_contract(
        SimCLRv2Spec(
            pretrain_epochs=0,
            finetune_epochs=0,
            distill_epochs=1,
            use_labeled_in_distill=False,
            student_bundle=_make_bundle(),
        ),
        SimCLRv2Method.info.capabilities,
        SimCLRv2Method.info.model_binding,
    )
    distill_student = next(
        component for component in distill_only.components if component.slot == "student_bundle"
    )
    assert "fit.X_l" not in distill_student.input_roles


def test_simclr_v2_loss_helpers() -> None:
    z = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = simclr_v2._nt_xent_loss(z, temperature=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None

    with pytest.raises(InductiveValidationError, match="temperature must be > 0"):
        simclr_v2._nt_xent_loss(z.detach(), temperature=0.0)
    with pytest.raises(InductiveValidationError, match="Projection outputs must be 2D"):
        simclr_v2._nt_xent_loss(torch.ones(2), temperature=0.5)
    with pytest.raises(InductiveValidationError, match="Contrastive batch must be even"):
        simclr_v2._nt_xent_loss(torch.ones((3, 2)), temperature=0.5)

    logits_s = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    logits_t = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    loss_detached = simclr_v2._distill_loss(logits_s, logits_t, temperature=1.0, detach_target=True)
    loss_detached.backward(retain_graph=True)
    assert logits_t.grad is None

    logits_s.grad = None
    logits_t.grad = None
    loss_attached = simclr_v2._distill_loss(
        logits_s, logits_t, temperature=1.0, detach_target=False
    )
    loss_attached.backward()
    assert logits_s.grad is not None
    assert logits_t.grad is not None

    with pytest.raises(InductiveValidationError, match="distill_temperature must be > 0"):
        simclr_v2._distill_loss(
            logits_s.detach(), logits_t.detach(), temperature=0.0, detach_target=True
        )


def test_simclr_v2_fit_pretrain_only_uses_unlabeled_fallback() -> None:
    base = make_torch_dataset()
    data = InductiveDataset(X_l=base.X_l, y_l=base.y_l, X_u=base.X_u, X_u_w=None, X_u_s=None)
    bundle = _make_projection_bundle()
    spec = _make_spec(
        pretrain_bundle=None,
        finetune_bundle=bundle,
        pretrain_epochs=1,
        finetune_epochs=0,
        distill_epochs=0,
        transfer_pretrain=False,
    )

    method = SimCLRv2Method(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._bundle is bundle
    assert method._backend == "torch"


def test_simclr_v2_fit_pretrain_finetune_and_distill_without_student() -> None:
    data = make_torch_ssl_dataset()
    pretrain_bundle = _make_projection_bundle()
    finetune_bundle = make_model_bundle()
    spec = _make_spec(
        pretrain_bundle=pretrain_bundle,
        finetune_bundle=finetune_bundle,
        student_bundle=None,
        pretrain_epochs=1,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=True,
        transfer_pretrain=True,
    )

    method = SimCLRv2Method(spec).fit(data, device=DeviceSpec(device="cpu"), seed=1)
    assert method._bundle is finetune_bundle
    assert method.device == "cpu"

    proba = method.predict_proba(data.X_l)
    assert proba.shape == (4, 2)
    assert torch.allclose(proba.sum(dim=1), torch.ones(4), atol=1e-6)


def test_simclr_v2_fit_distill_with_student_bundle_without_labeled_loss() -> None:
    data = make_torch_ssl_dataset()
    finetune_bundle = make_model_bundle()
    student_bundle = make_model_bundle()
    spec = _make_spec(
        pretrain_bundle=None,
        finetune_bundle=finetune_bundle,
        student_bundle=student_bundle,
        pretrain_epochs=0,
        finetune_epochs=0,
        distill_epochs=1,
        use_labeled_in_distill=False,
    )

    method = SimCLRv2Method(spec).fit(data, device=DeviceSpec(device="cpu"), seed=2)
    assert method._bundle is student_bundle


def test_simclr_v2_fit_data_errors() -> None:
    spec = _make_spec()
    with pytest.raises(InductiveValidationError, match="data must not be None"):
        SimCLRv2Method(spec).fit(None, device=DeviceSpec(device="cpu"))

    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        SimCLRv2Method(spec).fit(make_numpy_dataset(), device=DeviceSpec(device="cpu"))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"pretrain_epochs": -1}, "pretrain_epochs must be >= 0"),
        ({"finetune_epochs": -1}, "finetune_epochs must be >= 0"),
        ({"distill_epochs": -1}, "distill_epochs must be >= 0"),
        (
            {"pretrain_epochs": 0, "finetune_epochs": 0, "distill_epochs": 0},
            "At least one of pretrain_epochs",
        ),
        ({"alpha": -0.1}, "alpha must be in \\[0, 1\\]"),
        ({"alpha": 1.1}, "alpha must be in \\[0, 1\\]"),
        ({"temperature": 0.0}, "temperature must be > 0"),
        ({"distill_temperature": 0.0}, "distill_temperature must be > 0"),
    ],
)
def test_simclr_v2_fit_spec_validation_errors(overrides, match) -> None:
    with pytest.raises(InductiveValidationError, match=match):
        SimCLRv2Method(_make_spec(**overrides)).fit(
            make_torch_ssl_dataset(), device=DeviceSpec(device="cpu"), seed=0
        )


def test_simclr_v2_fit_dataset_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    base = make_torch_ssl_dataset()

    no_u = InductiveDataset(X_l=base.X_l, y_l=base.y_l, X_u=None, X_u_w=None, X_u_s=None)
    with pytest.raises(InductiveValidationError, match="requires unlabeled data"):
        SimCLRv2Method(_make_spec()).fit(no_u, device=DeviceSpec(device="cpu"), seed=0)

    empty_u = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u=base.X_u[:0],
        X_u_w=None,
        X_u_s=None,
    )
    with pytest.raises(InductiveValidationError, match="X_u must be non-empty"):
        SimCLRv2Method(_make_spec(pretrain_epochs=1, finetune_epochs=0, distill_epochs=0)).fit(
            empty_u, device=DeviceSpec(device="cpu"), seed=0
        )

    mismatch_u = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u=base.X_u,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s[:-1],
    )
    monkeypatch.setattr(simclr_v2, "ensure_torch_data", lambda data, device: data)
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        SimCLRv2Method(_make_spec(pretrain_epochs=1, finetune_epochs=0, distill_epochs=0)).fit(
            mismatch_u, device=DeviceSpec(device="cpu"), seed=0
        )

    empty_l = InductiveDataset(
        X_l=base.X_l[:0],
        y_l=base.y_l[:0],
        X_u=base.X_u,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        SimCLRv2Method(_make_spec(pretrain_epochs=0, finetune_epochs=1, distill_epochs=0)).fit(
            empty_l, device=DeviceSpec(device="cpu"), seed=0
        )

    bad_dtype = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l.to(dtype=torch.int32),
        X_u=base.X_u,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="y_l must be int64"):
        SimCLRv2Method(_make_spec(pretrain_epochs=0, finetune_epochs=1, distill_epochs=0)).fit(
            bad_dtype, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(
        InductiveValidationError, match="pretrain_bundle or finetune_bundle must be provided"
    ):
        SimCLRv2Method(
            _make_spec(
                pretrain_bundle=None,
                finetune_bundle=None,
                pretrain_epochs=1,
                finetune_epochs=0,
                distill_epochs=0,
            )
        ).fit(base, device=DeviceSpec(device="cpu"), seed=0)

    with pytest.raises(
        InductiveValidationError, match="finetune_bundle or pretrain_bundle must be provided"
    ):
        SimCLRv2Method(
            _make_spec(
                pretrain_bundle=None,
                finetune_bundle=None,
                pretrain_epochs=0,
                finetune_epochs=1,
                distill_epochs=0,
            )
        ).fit(base, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_fit_pretrain_projection_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = make_torch_ssl_dataset()
    spec = _make_spec(pretrain_epochs=1, finetune_epochs=0, distill_epochs=0)
    _patch_single_step(monkeypatch)

    outputs = _iter_outputs(torch.zeros(2), torch.zeros(2))
    monkeypatch.setattr(simclr_v2, "_forward_projection", lambda *_args, **_kwargs: next(outputs))
    with pytest.raises(InductiveValidationError, match="Projection outputs must be 2D"):
        SimCLRv2Method(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    outputs = _iter_outputs(torch.zeros((2, 2)), torch.zeros((2, 3)))
    monkeypatch.setattr(simclr_v2, "_forward_projection", lambda *_args, **_kwargs: next(outputs))
    with pytest.raises(InductiveValidationError, match="same shape"):
        SimCLRv2Method(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_fit_finetune_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    data = make_torch_ssl_dataset()
    spec = _make_spec(pretrain_epochs=0, finetune_epochs=1, distill_epochs=0)
    _patch_single_step(monkeypatch)

    monkeypatch.setattr(simclr_v2, "_forward_logits", lambda *_args, **_kwargs: torch.zeros(2))
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        SimCLRv2Method(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    monkeypatch.setattr(simclr_v2, "_forward_logits", lambda *_args, **_kwargs: torch.zeros((2, 2)))
    bad = InductiveDataset(
        X_l=data.X_l,
        y_l=torch.tensor([0, 2, 1, 1], dtype=torch.int64),
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="labels must be within"):
        SimCLRv2Method(spec).fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_fit_distill_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    data = make_torch_ssl_dataset()
    _patch_single_step(monkeypatch)

    outputs = _iter_outputs(torch.zeros(2), torch.zeros((2, 2)))
    monkeypatch.setattr(simclr_v2, "_forward_logits", lambda *_args, **_kwargs: next(outputs))
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        SimCLRv2Method(
            _make_spec(
                pretrain_epochs=0,
                finetune_epochs=0,
                distill_epochs=1,
                use_labeled_in_distill=False,
                student_bundle=make_model_bundle(),
            )
        ).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    outputs = _iter_outputs(torch.zeros((2, 2)), torch.zeros((2, 3)))
    monkeypatch.setattr(simclr_v2, "_forward_logits", lambda *_args, **_kwargs: next(outputs))
    with pytest.raises(InductiveValidationError, match="shape mismatch"):
        SimCLRv2Method(
            _make_spec(
                pretrain_epochs=0,
                finetune_epochs=0,
                distill_epochs=1,
                use_labeled_in_distill=False,
                student_bundle=make_model_bundle(),
            )
        ).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    outputs = _iter_outputs(torch.zeros((2, 2)), torch.zeros((2, 2)), torch.zeros(2))
    monkeypatch.setattr(simclr_v2, "_forward_logits", lambda *_args, **_kwargs: next(outputs))
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        SimCLRv2Method(_make_spec(pretrain_epochs=0, finetune_epochs=0, distill_epochs=1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    outputs = _iter_outputs(
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
    )
    monkeypatch.setattr(simclr_v2, "_forward_logits", lambda *_args, **_kwargs: next(outputs))
    bad = InductiveDataset(
        X_l=data.X_l,
        y_l=torch.tensor([0, 2, 1, 1], dtype=torch.int64),
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="labels must be within"):
        SimCLRv2Method(_make_spec(pretrain_epochs=0, finetune_epochs=0, distill_epochs=1)).fit(
            bad, device=DeviceSpec(device="cpu"), seed=0
        )


def test_simclr_v2_predict_proba_errors() -> None:
    method = SimCLRv2Method()
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(torch.zeros((2, 2)))

    method._bundle = _make_bundle()
    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        method.predict_proba(np.zeros((2, 2), dtype=np.float32))

    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="requires torch.Tensor or dict inputs"):
        method.predict_proba([[0.0, 1.0]])

    method._bundle = _make_bundle(_BadLogits1D())
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method.predict_proba(torch.zeros((2, 2)))


def test_simclr_v2_predict_proba_tensor_and_dict_inputs() -> None:
    method = SimCLRv2Method()
    bundle = _make_bundle(_MetaNet())
    method._bundle = bundle
    method._backend = None

    bundle.model.train()
    proba = method.predict_proba(torch.zeros((2, 2), dtype=torch.float32))
    assert proba.shape == (2, 2)
    assert bundle.model.training is True

    bundle.model.eval()
    proba_dict = method.predict_proba({"x": torch.zeros((0, 2), dtype=torch.float32)})
    assert proba_dict.shape == (0, 2)
    assert bundle.model.training is False


def test_simclr_v2_predict_proba_batches_large_inputs() -> None:
    method = SimCLRv2Method(SimCLRv2Spec(batch_size=2))
    model = _CountingMetaNet()
    method._bundle = _make_bundle(model)
    method._backend = "torch"

    proba = method.predict_proba(torch.zeros((5, 2), dtype=torch.float32))
    assert proba.shape == (5, 2)
    assert model.batch_sizes == [2, 2, 1]


def test_simclr_v2_predict_proba_dict_batches_and_empty_error_paths() -> None:
    method = SimCLRv2Method(SimCLRv2Spec(batch_size=2))
    model = _CountingMetaNet()
    method._bundle = _make_bundle(model)
    method._backend = "torch"

    proba = method.predict_proba({"x": torch.zeros((3, 2), dtype=torch.float32)})
    assert proba.shape == (3, 2)
    assert model.batch_sizes == [2, 1]

    bad = SimCLRv2Method(SimCLRv2Spec(batch_size=2))
    bad._bundle = _make_bundle(_BadLogits1D())
    bad._backend = "torch"
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        bad.predict_proba(torch.zeros((0, 2), dtype=torch.float32))
