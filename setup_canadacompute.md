# Canada Compute (Digital Research Alliance) – Quick Setup Guide

## 1. Request an Account

1. Go to: [https://alliancecan.ca](https://alliancecan.ca)
2. Create an account (CCDB).
3. Join a **PI / Sponsor project** (or request a RAC if applicable).
4. Request access to systems:

   * Narval (GPU)
   * Narval (GPU/CPU)
   * Vulcan (GPU)
   * HPSS (storage)
   * (Optional) Narval Cloud / Alliance Cloud (VMs)
5. Wait for approval emails.

---

## 2. Generate SSH Key (Local Machine)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Start SSH agent and add key:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

Copy public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Add it to the Alliance portal → **CCDB → Manage SSH Keys**

---

## 3. SSH Config

Create `~/.ssh/config`:

```bash
Host narval
    HostName narval.alliancecan.ca
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlMaster auto
    ControlPersist 1d

Host vulcan
    HostName vulcan.alliancecan.ca
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlMaster auto
    ControlPersist 1d
```

Connect:

```bash
ssh narval
ssh vulcan
```

---

## 4. Important Directories

| Directory  | Use                 |
| ---------- | ------------------- |
| `$HOME`    | Code, small files   |
| `$SCRATCH` | Training data, jobs |
| `$PROJECT` | Shared project data |
| HPSS       | Archive storage     |

**Do not store important data in SCRATCH (it gets purged).**

---

## 5. Transfer Files (CLI)

Use `rsync` (recommended):

```bash
rsync -av ./project/ username@narval.alliancecan.ca:$SCRATCH/project/
```

Download results:

```bash
rsync -av username@narval.alliancecan.ca:$SCRATCH/project/output/ ./output/
```

For large datasets → use **Globus**. See [here](./using_globus.md)

---

## 6. Python Environment (3.12 — local and cluster)

This repo targets **Python 3.12** everywhere so local checks match Slurm jobs. If a cluster does not expose `python/3.12`, run `module spider python` and pick the closest **3.12.x** module name.

### Local machine (repo `.venv`)

```bash
cd /path/to/COMP-767
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Cluster (login node)

Create a virtual environment **once** (on the login node is fine; you need network for `pip`). Keep the venv under `$HOME` so it is not wiped with SCRATCH.

```bash
module load python/3.12
mkdir -p ~/venvs
virtualenv --no-download ~/venvs/comp767
source ~/venvs/comp767/bin/activate
pip install --upgrade pip
```

From your project root (where `requirements.txt` lives), install this repo’s dependencies:

```bash
cd $SCRATCH/comp-767   # or wherever you cloned the repo
pip install -r requirements.txt
```

GPU jobs need a **CUDA-enabled** PyTorch build. If `import torch` works but `torch.cuda.is_available()` is false on a GPU node, reinstall PyTorch using the [official install selector](https://pytorch.org/get-started/locally/) (pick Linux + Pip + a CUDA version that matches the cluster). Example pattern (adjust CUDA index URL to match Narval’s stack when in doubt):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Sanity check inside an interactive GPU allocation:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

**Slurm jobs** must load the same modules and **activate the same venv** before calling `python` (see `scratch/dummy_test.slurm`).

### Hugging Face and PyTorch caches → `$SCRATCH`

Model weights (often **10+ GB**) should **not** fill `$HOME`. In Slurm scripts (and interactive GPU shells), point caches at scratch **before** importing `transformers` or `torch.hub`:

```bash
export HF_HOME="${SCRATCH}/huggingface"
export TORCH_HOME="${SCRATCH}/torch"
export TMPDIR="${SCRATCH}/tmp/${SLURM_JOB_ID:-interactive}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
```

With `HF_HOME` set, Hugging Face Hub models go under `${HF_HOME}/hub` by default. `TORCH_HOME` covers `torch.hub` / related downloads. `TMPDIR` avoids large temp files on small local disks.

---

## 7. Running Jobs (SLURM)

### Interactive job (testing)

```bash
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00
export HF_HOME="${SCRATCH}/huggingface"
export TORCH_HOME="${SCRATCH}/torch"
export TMPDIR="${SCRATCH}/tmp/${SLURM_JOB_ID:-interactive}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
module load python/3.12
source ~/venvs/comp767/bin/activate
python train.py
```

### Batch job (`train.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

export HF_HOME="${SCRATCH}/huggingface"
export TORCH_HOME="${SCRATCH}/torch"
export TMPDIR="${SCRATCH}/tmp/${SLURM_JOB_ID:-job}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

module load python/3.12
source ~/venvs/comp767/bin/activate
cd $SCRATCH/project
python train.py
```

Submit:

```bash
sbatch train.slurm
```

Monitor:

```bash
squeue -u $USER
squeue --start -u $USER
```

---

## 8. Rules (Important)

* Do **not** run heavy jobs on login node
* Use **SLURM** for compute
* Save checkpoints (jobs can crash)
* SCRATCH is temporary
* HOME/PROJECT backed up
* Use GPUs via `--gres=gpu`

---

## 9. Typical Workflow

```bash
# Local
git push
rsync project -> cluster

# Cluster
ssh narval
cd $SCRATCH/project
sbatch train.slurm

# After training
rsync results -> local
```

---

## 10. Useful Commands

| Command        | Purpose            |
| -------------- | ------------------ |
| `ssh`          | Connect to cluster |
| `rsync`        | File transfer      |
| `salloc`       | Interactive job    |
| `sbatch`       | Submit job         |
| `squeue`       | View jobs          |
| `scancel`      | Cancel job         |
| `module avail` | List software      |
| `nvidia-smi`   | Check GPU          |
| `htop`         | Check CPU          |

---

## 11. Recommended Systems

| System       | Use                |
| ------------ | ------------------ |
| Narval       | Large GPU jobs     |
| Narval       | General GPU jobs   |
| Vulcan       | Small GPU jobs     |
| HPSS         | Long-term storage  |
| Alliance Cloud | VM / hosting / dev |

---

This is the standard workflow used for ML training on Digital Research Alliance clusters.
