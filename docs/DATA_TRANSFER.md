# Torch to Cloud Bursting data reproduction

Use a CPU compute allocation. Torch and Bursting compute storage are separate.
Keep the interactive allocation active until transfers and validation finish.
No GPU job is required. An interrupted transfer can be rerun; it does not delete
source or destination files. Never run two copies for the same destination.

After completing NYU SSH/Microsoft authentication with a ControlMaster socket:

```bash
TORCH_SOCKET=/home/yy5919/.ssh/pyro-torch-cpu-control \
  bash hpc/pull_torch_datasets.sh > outputs/transfers/pull_torch_datasets.log 2>&1
```

The script checks authentication, takes the shared exclusive transfer lock,
and copies MTG, EgoSchema and REFRAMED serially. Defaults match this run.
TORCH_HOST, TORCH_SOURCE, TRANSFER_DEST and TRANSFER_BWLIMIT may be overridden.
The limit is KiB/s; default 100000. Do not restart a live copy simply because
its observation handle times out. Check the process and current log first.
The initial run uses the equivalent preserved script under outputs/transfers;
this repository script has not been launched over that running transfer.

After successful transfer, verify MTG hashes:

```bash
(cd /scratch/yy5919/mtg-ad-transfer/mtg-ad && sha256sum -c SHA256SUMS)
```

Run the repository verification script for each evaluation dataset, using the
existing CPU container environment when ffprobe is not on host PATH:

```bash
source hpc/config.sh
in_env ro 'python scripts/verify_transferred_dataset.py --root /scratch/yy5919/mtg-ad-transfer/Evaluation-datasets/egoschema --output outputs/verification/egoschema-transfer-final.json --probe'
in_env ro 'python scripts/verify_transferred_dataset.py --root /scratch/yy5919/mtg-ad-transfer/Evaluation-datasets/reframed --output outputs/verification/reframed-transfer-final.json --probe'
```

Checks cover manifest IDs, sizes, source snapshot totals, stream types and
container durations. They do not claim full decode or evaluation-video hashes.
Source-missing REFRAMED videos remain missing; never create placeholders.

Prepare loader-compatible views (safe to rerun):

```bash
in_env ro 'python scripts/prepare_dataset_views.py --source-root /scratch/yy5919/mtg-ad-transfer --view-root /scratch/yy5919/pyro-kd-derived/datasets'
```

Use that view root for loaders. REFRAMED's view points to the preserved encrypted
source annotations; its separate keys directory contains the published dataset
decryption keys. Raw files are unchanged. Derived frame/audio caches belong
under /scratch/yy5919/pyro-kd-derived/cache, outside the raw source tree.

Official annotation counts: MTG 332 validation / 101 test; REFRAMED 1836 train /
96 validation / 91 test; EgoSchema 5031 questions / 500 public labels. Available
video counts: MTG 413; REFRAMED 1983; EgoSchema 5031. Availability is distinct from
model results. Keep the 20 MTG and 40 REFRAMED source failures in reports.
