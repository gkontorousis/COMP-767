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

Host vulcan
    HostName vulcan.alliancecan.ca
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_ed25519
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

## 6. Python Environment (On Cluster)

```bash
module load python/3.10
virtualenv --no-download ~/venvs/myenv
source ~/venvs/myenv/bin/activate
pip install --upgrade pip
pip install torch torchvision
```

---

## 7. Running Jobs (SLURM)

### Interactive job (testing)

```bash
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00
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

module load python/3.10
source ~/venvs/myenv/bin/activate
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
