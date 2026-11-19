# NYU HPC Cloud Bursting (course cluster), as documented in hpc/README.md.
# Override any value in hpc/env.sh. Partitions are per-GPU-type on this cluster,
# so GPU jobs choose a partition rather than a --gres type.
export PYRO_PROFILE=cloud_bursting
export CPU_PARTITION="${CPU_PARTITION:-n2c48m24}"
export GPU_PARTITION="${GPU_PARTITION:-c12m85-a100-1}"      # 1x A100 40 GB
export GPU_PARTITION_2="${GPU_PARTITION_2:-c24m170-a100-2}" # 2x A100 40 GB
export GPU_GRES="${GPU_GRES:-gpu:1}"
export PYRO_ENV_MODE="${PYRO_ENV_MODE:-container}"
