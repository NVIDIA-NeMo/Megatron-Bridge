# Shared Slurm container backends

The public training, conversion and inference launchers default to **Pyxis**.
On sites where Pyxis forces its runtime onto unsuitable storage, select
`--container-runtime enroot --enroot-root /path/to/owned/lustre/runtime`.
The equivalent defaults are `BRIDGE_CONTAINER_RUNTIME=enroot` and
`BRIDGE_ENROOT_ROOT`; these can stay in private deployment configuration rather
than model-verification commands. Local conversion does not use this option.

Direct Enroot requires:

- An existing cluster-side `.sqsh` or `.squashfs` image supplied through
  `--container-image` or `CONTAINER_IMAGE`. No registry pull, import, copy, or
  image build is performed by this backend.
- An existing, writable, user-owned runtime directory on Lustre.
- `NEMORUN_HOME`, the image and all explicit mount sources on Lustre. The
  submitting-node checks run before creating the experiment; allocation-side
  checks repeat them before starting the container.
- Enroot on every allocated node, supporting `enroot info` and starting a
  SquashFS file directly (including its site-provided SquashFUSE dependencies).
- A working container installation of Bridge or an explicit source mount such
  as `--mount /path/to/lustre/source:/opt/Megatron-Bridge:ro`. Mounts accept
  `HOST:CONTAINER[:ro|rw]`. The backend does not package a source checkout into
  the image or install dependencies.

The generated job uses plain `srun` without Pyxis container flags. Each Slurm
task creates a private job/rank-specific directory under the selected root,
sets all five Enroot data/runtime/cache/temp/config paths, then checks their
effective values with the installed Enroot before starting the image. Site
overrides fail the task; there is no silent fallback to Pyxis. Home mounting
and root-filesystem writes are disabled.

Common application caches default to subdirectories of that runtime directory.
Explicitly forwarded cache variables must name existing Lustre directories;
the backend mounts them at the same paths. Container `/tmp` and `/var/tmp` are
Lustre-backed bind mounts, keeping Unix socket paths short. Source bytecode
writes are disabled. Model/data/output paths and application-specific cache or
build settings still belong to the caller; ensure those also use explicit
Lustre mounts. Configure the submitting shell's `UV_CACHE_DIR` and `TMPDIR` on
Lustre **before** invoking the public shell launcher, which initializes its
Python environment before parsing backend options.

Environment forwarding remains name-only (`--env NAME`): credential values
are not embedded in the generated script. Worker arguments, Slurm task/rank
metadata, resource requests and exit codes are preserved. Existing synchronous
waiting still polls no faster than once per minute. Runtime artifacts are
retained for diagnosis; this backend performs no recursive cleanup.

Use `--submission-dry-run` to inspect the generated job before submission.
This validates rendering, not the installed Enroot runtime or model workload.
