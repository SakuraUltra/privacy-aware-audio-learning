"""Dataset-free model checks used by the existing unified entrypoint."""


def run_model_smoke_test(feature_type, mode):
    """Exercise a small CPU model; this is not a dataset or privacy evaluation."""
    import torch
    import torch.nn.functional as F
    from configs.config_factory import ConfigFactory
    from models.transformer import TransformerClassifier
    from models.vib_model import build_vib_model

    if mode not in ("normal", "vib"):
        raise ValueError("Synthetic model checks support normal/vib only")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = ConfigFactory.create_config(feature_type, mode)
    config.general["device"] = torch.device("cpu")
    config.model.update(num_layers=1, dim_feedforward=64, dropout=0.0)
    if mode == "vib":
        model, _, _ = build_vib_model(config, config.vib_cfg.__dict__, steps_per_epoch=1)
    else:
        model = TransformerClassifier(**config.model, dp_mode=False)
    model = model.cpu()
    features = torch.randn(4, 12, config.get_model_input_dim())
    labels = torch.tensor([0, 1, 0, 1])
    mask = torch.zeros(4, 12, dtype=torch.bool)
    mask[0, 9:] = True
    features[mask] = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.train()
    optimizer.zero_grad()
    outputs = model(features, padding_mask=mask)
    logits = outputs[0]
    if logits.shape != (4, 2) or not torch.isfinite(logits).all():
        raise RuntimeError("Unexpected or non-finite classifier outputs")
    loss = F.cross_entropy(logits, labels)
    if mode == "vib":
        loss = loss + config.vib_params["beta"] * outputs[2]["kl"]
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite synthetic loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing or non-finite gradients")
    if not any(torch.count_nonzero(g).item() for g in gradients):
        raise RuntimeError("All gradients are zero")
    optimizer.step()
    if not any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()):
        raise RuntimeError("Optimizer did not update model parameters")
    model.eval()
    with torch.no_grad():
        prediction = model(features, padding_mask=mask)[0]
    if prediction.shape != (4, 2) or not torch.isfinite(prediction).all():
        raise RuntimeError("Invalid evaluation outputs")
    print(f"PASS: {feature_type}/{mode} on CPU — forward, backward, optimizer update, eval")
    print("Synthetic features only; no audio, checkpoints, network downloads, or benchmark metrics.")
