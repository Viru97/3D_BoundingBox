import torch

from sereact_3d_bbox.data.splits import create_split_manifest, split_scene_ids
from sereact_3d_bbox.inference import load_model
from sereact_3d_bbox.metrics import cuboid_corner_permutations
from sereact_3d_bbox.models.dgcnn import DGCNNBBoxV2, rotation_6d_to_matrix
from sereact_3d_bbox.models.loss import BBoxLoss, cuboid_permutation_loss, matched_pose_losses

from .conftest import make_dataset


def test_group_disjoint_split_manifest(tmp_path):
    data_root = make_dataset(tmp_path, scenes=10)
    manifest = create_split_manifest(data_root, tmp_path / "splits.json", seed=7)
    train = set(split_scene_ids(manifest, "train"))
    val = set(split_scene_ids(manifest, "val"))
    test = set(split_scene_ids(manifest, "test"))
    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_decoder_rotation_is_orthonormal():
    rot = torch.randn(5, 6)
    matrix = rotation_6d_to_matrix(rot)
    eye = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(torch.bmm(matrix.transpose(1, 2), matrix), eye, atol=1e-5)


def test_cuboid_loss_is_invariant_to_valid_corner_permutation():
    corners = torch.tensor(
        [[[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]],
        dtype=torch.float32,
    )
    perm = cuboid_corner_permutations()[3]
    assert cuboid_permutation_loss(corners, corners[:, perm]) < 1e-6


def test_pose_losses_are_small_for_exact_box():
    center = torch.zeros(1, 3)
    log_dims = torch.log(torch.tensor([[2.0, 2.0, 2.0]]))
    rot6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    model = DGCNNBBoxV2(in_channels=10, k=4)
    corners = model.get_3d_box(center, log_dims, rot6d)
    losses = matched_pose_losses(corners, corners, log_dims, rot6d)
    assert losses["corner"] < 1e-6
    assert losses["dimension"] < 1e-6
    assert losses["rotation"] < 1e-6


def test_bbox_loss_reports_pose_components():
    model = DGCNNBBoxV2(in_channels=10, k=4)
    points = torch.randn(2, 10, 32)
    center, log_dims, rot6d = model(points)
    pred = model.get_3d_box(center, log_dims, rot6d)
    criterion = BBoxLoss()
    total, losses = criterion(pred, pred.detach(), center, log_dims, rot6d)
    assert total.isfinite()
    assert {"corner", "dimension", "rotation", "rotation_regularizer"}.issubset(losses)


def test_checkpoint_load_roundtrip(tmp_path):
    model = DGCNNBBoxV2(in_channels=10, k=4)
    ckpt = tmp_path / "model.pth"
    torch.save({"model": model.state_dict(), "args": {"model_version": "v2", "in_channels": 10, "k_neighbors": 4}}, ckpt)
    loaded, kwargs = load_model(ckpt, torch.device("cpu"))
    out = loaded(torch.randn(2, 10, 32))
    assert kwargs["version"] == "v2"
    assert out[0].shape == (2, 3)
