import torch
from torch.utils.data import DataLoader

from sereact_3d_bbox.data.dataset import PointCloudInstanceDataset, collate_fn
from sereact_3d_bbox.data.splits import create_split_manifest, split_scene_ids
from sereact_3d_bbox.inference import NpyMaskProvider, predict_sample
from sereact_3d_bbox.models.dgcnn import DGCNNBBoxV2
from sereact_3d_bbox.models.loss import BBoxLoss

from .conftest import make_dataset


def test_synthetic_batch_trains_and_infers(tmp_path):
    data_root = make_dataset(tmp_path, scenes=10)
    manifest = create_split_manifest(data_root, tmp_path / "splits.json", seed=11)
    train_ds = PointCloudInstanceDataset(data_root, num_points=32, is_train=True, scene_ids=split_scene_ids(manifest, "train"), seed=11)
    loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = DGCNNBBoxV2(in_channels=10, k=4, dropout=0.0)
    criterion = BBoxLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        center, log_dims, rot6d = model(batch["points"])
        pred = model.get_3d_box(center, log_dims, rot6d)
        loss, _ = criterion(pred, batch["target"], center, log_dims, rot6d, batch["target_center"], batch["target_dims"])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] <= losses[0] * 1.2

    model.eval()
    predictions = predict_sample(model, torch.device("cpu"), data_root / split_scene_ids(manifest, "test")[0], NpyMaskProvider(), num_points=32)
    assert len(predictions) == 1
    assert predictions[0].corners.shape == (8, 3)
