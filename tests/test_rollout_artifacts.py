import numpy as np

from hvp_rev.provenance import ArtifactStore, RunSpec
from hvp_rev.rollout_artifacts import write_complete_rollout


def test_complete_rollout_writes_aligned_record_metrics(tmp_path):
    spec = RunSpec(
        study="rollout", arm="fixture", seed=0, dataset_hash="d", ae_hash="a",
        arm_config={}, train_examples=1, optimizer_updates=1, batch_size=1, precision="fp32",
        nfe=1, warmup_frames=1, rollout_frames=2, evaluation_clips=1,
        source_commit="fixture", environment={"fixture": True},
    )
    store = ArtifactStore(tmp_path, spec)
    gt = np.zeros((1, 2, 1, 8, 8), np.float32)
    pred = gt.copy()
    gt[:, :, :, 2:6, 2:6] = 1
    path = write_complete_rollout(
        store, pred=pred, true=gt, clip=np.array([0, 0]), horizon=np.array([1, 2]),
        threshold=0.5, labels={"dt": np.array([9.0, 10.0])},
    )
    with np.load(path) as data:
        assert data["mse_full"].shape == (2,)
        assert data["fg_laplacian_energy_nonempty"].shape == (2,)
        assert data["clip"].tolist() == [0, 0]
        assert data["dt"].tolist() == [9.0, 10.0]
