"""Four-process Accelerate smoke witness for authoritative exp05 training."""

from accelerate import Accelerator
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def main() -> None:
    """Prove that all four ranks step and finish with synchronized parameters."""
    accelerator = Accelerator(mixed_precision="bf16")
    if accelerator.num_processes != 4:
        raise RuntimeError(
            f"DDP smoke requires 4 ranks, got {accelerator.num_processes}"
        )
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16) / 1024.0
    targets = features.mean(dim=1, keepdim=True)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=1,
        shuffle=False,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    local_steps = torch.zeros(1, device=accelerator.device)
    for inputs, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - labels).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        local_steps += 1
    gathered_steps = accelerator.gather(local_steps)
    flat = torch.cat(
        [
            parameter.detach().reshape(-1)
            for parameter in accelerator.unwrap_model(model).parameters()
        ]
    )
    checksum = flat.sum().view(1).to(accelerator.device)
    gathered_checksums = accelerator.gather(checksum)
    if accelerator.is_main_process:
        assert gathered_steps.shape == (4,)
        assert (gathered_steps > 0).all()
        assert torch.allclose(
            gathered_checksums,
            gathered_checksums[0].expand_as(gathered_checksums),
        )
        print("DDP_SMOKE_OK world_size=4 all_ranks_active=1 parameters_synced=1")


if __name__ == "__main__":
    main()
