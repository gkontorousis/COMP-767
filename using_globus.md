# Globus CLI Guide (Alliance / Compute Canada)

## Overview

Globus is a high-speed file transfer system used by HPC centers. It is more reliable than `scp` or `rsync` for large transfers and supports resumable background transfers.

Use Globus when:

* Files > 10GB
* Many files
* Transfers fail with rsync
* You want background transfers

---

## 1. Install Globus CLI

### Mac

```bash
brew install globus-cli
```

### Linux

```bash
pip install globus-cli
```

Verify:

```bash
globus version
```

---

## 2. Login

```bash
globus login
```

This opens a browser → login with your institution / Digital Research Alliance account.

---

## 3. Find Alliance Endpoints

Common endpoints:

* `computecanada#beluga-dtn`
* `computecanada#narval-dtn`
* `computecanada#vulcan-dtn`
* `computecanada#hpss`

Search:

```bash
globus endpoint search computecanada
```

---

## 4. Get Your Local Endpoint ID

```bash
globus endpoint local-id
```

Start local endpoint if needed:

```bash
globus endpoint start <LOCAL_ENDPOINT_ID>
```

---

## 5. Transfer Files

### Local → Cluster

```bash
globus transfer \
LOCAL_ENDPOINT_ID:/home/user/project \
computecanada#beluga-dtn:/scratch/username/project \
--recursive
```

### Cluster → Local

```bash
globus transfer \
computecanada#beluga-dtn:/scratch/username/output \
LOCAL_ENDPOINT_ID:/home/user/output \
--recursive
```

---

## 6. Monitor Transfers

```bash
globus task list
globus task show TASK_ID
```

Cancel transfer:

```bash
globus task cancel TASK_ID
```

---

## 7. Typical Workflow

```bash
# Login
globus login

# Get local endpoint
globus endpoint local-id

# Upload dataset
globus transfer \
LOCAL_ID:/Users/me/datasets \
computecanada#narval-dtn:/scratch/username/datasets \
--recursive

# Download results
globus transfer \
computecanada#narval-dtn:/scratch/username/output \
LOCAL_ID:/Users/me/output \
--recursive
```

---

## 8. When to Use Which Tool

| Tool   | Use               |
| ------ | ----------------- |
| scp    | Small files       |
| rsync  | Code, medium data |
| Globus | Large datasets    |
| HPSS   | Archive storage   |

---

## 9. Recommended Usage

| File Type   | Tool   |
| ----------- | ------ |
| Code        | rsync  |
| Dataset     | Globus |
| Checkpoints | Globus |
| Logs        | rsync  |
| Archive     | HPSS   |

---

## Notes

* Always transfer using **DTN (data transfer nodes)**: `*-dtn`
* Transfer datasets to `$SCRATCH`
* Move final results to `$PROJECT` or local machine
* Use HPSS for long-term storage

---

## Linking Markdown Files Locally

Yes, you can link one `.md` file to another locally using relative paths.

### Example folder structure:

```
project/
├── README.md
├── globus-guide.md
├── slurm-guide.md
```

### In `README.md`:

```md
## Guides

- [Globus Guide](./globus-guide.md)
- [SLURM Guide](./slurm-guide.md)
```

### You can also link to a section:

```md
[Go to Transfers Section](./globus-guide.md#5-transfer-files)
```

This works on:

* GitHub
* VS Code
* Most Markdown viewers

---

## Recommended Repo Structure

If you're going to do a lot of HPC/ML work:

```
compute-canada-guide/
├── README.md
├── ssh-setup.md
├── globus-guide.md
├── slurm-guide.md
├── pytorch-guide.md
└── vscode-remote.md
```

This becomes your personal HPC manual (very useful over time).
