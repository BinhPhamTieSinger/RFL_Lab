"""Read-only previews of the actual in-memory client partitions."""

import base64
import io

import torch
from PIL import Image

from rfl.client.fedhep import label_stats
from rfl.data.cifar100 import MEAN, STD


def describe_clients(cfg, clients):
    classes = 100 if cfg.data.dataset == "cifar100" else 10
    names = [str(i) for i in range(classes)]
    if cfg.data.dataset == "cifar100":
        from torchvision.datasets import CIFAR100

        names = CIFAR100(cfg.data.root, train=True, download=False).classes
    output = []
    for cid, client in enumerate(clients):
        counts = torch.bincount(client.y_train.cpu(), minlength=classes)
        skew, _, _ = label_stats(client.y_train, classes)
        samples = []
        if cfg.data.dataset == "cifar100":
            # First three distinct labels in partition order. This consumes no RNG.
            seen = set()
            for index, label in enumerate(client.y_train.cpu().tolist()):
                if label in seen:
                    continue
                seen.add(label)
                image = client.x_train[index].detach().cpu()
                image = image * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
                pixels = image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
                encoded = io.BytesIO()
                Image.fromarray(pixels).save(encoded, format="PNG")
                samples.append(
                    {
                        "label": label,
                        "name": names[label],
                        "partition_index": index,
                        "image": "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode(),
                    }
                )
                if len(samples) == 3:
                    break
        output.append(
            {
                "id": cid,
                "train": client.num_train,
                "test": len(client.y_test),
                "histogram": counts.tolist(),
                "class_names": names,
                "classes_present": int((counts > 0).sum()),
                "r_skew": skew,
                "samples": samples,
                "sample_scope": "Original training partition before attack transforms",
            }
        )
    return output
