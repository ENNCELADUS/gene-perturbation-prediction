"""Audited parameter provenance for response-only adaptation."""


def configure_parameters(backbone, loaded_keys, *, unfreeze=False):
    prefix = "state_adapter.state_model."
    new_state = "basal_encoder.0.weight"
    state_parameters = dict(backbone.state_adapter.state_model.named_parameters())
    if set(state_parameters) - set(loaded_keys) != {new_state}:
        raise ValueError("unexpected fresh STATE parameters")
    expected = {
        prefix + new_state: (328, 2560),
        "perturbations.adapter.net.0.weight": (512, 1280),
        "perturbations.adapter.net.0.bias": (512,),
        "perturbations.adapter.net.2.weight": (2024, 512),
        "perturbations.adapter.net.2.bias": (2024,),
    }
    actual = dict(backbone.named_parameters())
    if any(
        n not in actual or tuple(actual[n].shape) != shape
        for n, shape in expected.items()
    ):
        raise ValueError("P1-B interface shape mismatch")
    inherited = {prefix + n for n in state_parameters if n in loaded_keys}
    if set(actual) != inherited | set(expected):
        raise ValueError("unclassified backbone parameters")
    backbone.requires_grad_(False)
    backbone.eval()
    for name in expected:
        actual[name].requires_grad_(True)
    groups = [
        {"params": [actual[n] for n in expected], "lr": 1e-4, "name": "interface"}
    ]
    if unfreeze:
        for name in inherited:
            actual[name].requires_grad_(True)
        groups.append(
            {
                "params": [actual[n] for n in sorted(inherited)],
                "lr": 1e-6,
                "name": "inherited",
            }
        )
    report = {
        n: {
            "shape": list(p.shape),
            "origin": "new" if n in expected else "inherited",
            "trainable": p.requires_grad,
        }
        for n, p in actual.items()
    }
    return groups, report
