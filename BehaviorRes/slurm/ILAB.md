# iLab + Slurm cheat sheet (BehaviorRes)

Save this next to the submit scripts. Official Rutgers CS pages this is based on:

- [Limitations enforced on CS Linux machines](https://resources.cs.rutgers.edu/docs/limitation-enforced-on-cs-linux-machines/)
- [Scheduler for GPU and long CPU jobs](https://resources.cs.rutgers.edu/docs/scheduler-for-gpu-jobs/)

Your cluster copy of this project is:

```text
/common/users/so504/Desktop/BehaviorRes
```

Login nodes you have used: `ilab1`, `ilab3` (`ilab*.cs.rutgers.edu`). It does not matter which one you `ssh` into; Slurm places the job on a node that has free GPUs.

---

## 1. Why Slurm on iLab

Managed GPU machines (iLab1–4, rLab1–4, rLab6–7) **will not give you a GPU unless the job is started with `sbatch` or `srun`**. Outside Slurm, `nvidia-smi` can still list cards, but they are not yours. Use `nvidia-smi-priv` to peek from a login shell; inside a Slurm job, `nvidia-smi` shows only the GPU you were allocated.

Slurm jobs are **exempt** from the interactive CPU-hour and 80 GB iLab memory caps on that first page. Interactive `ssh` / `tmux` / `nohup` jobs are not.

CS policy: **big / long GPU work must go through the scheduler**. Interactive `srun` is for short tests. Batch (`sbatch`) is what this repo uses.

---

## 2. Machines, GPUs, and limits (CS policy)

**Slurm-managed nodes:** iLab1–4, rLab1–4, rLab6–7.

**Cards currently listed:** RTX A4000, A4500, A4500 ADA, A5000, A6000, RTX Pro 5000 Blackwell, NVIDIA A100 40 GB.

**Not in the scheduler:** iLab desktops (one GPU, first-come), `ilabU` (test Ubuntu box). Those are still under the interactive limits.

| Resource | Interactive (no Slurm) | Slurm job |
|---|---|---|
| GPU on iLab/rLab servers | none | request with `-G` / `--gpus` |
| CPU lifetime | `keep-job N` (N daytime hours); min 24 **CPU**-hours. 4 cores → ~6 wall hours to hit 24 | not those caps; job may run up to **7 days**, then can be killed if others need GPUs |
| Memory on `ilab*` | 80 GB / user | default **40 GB**; you may ask up to ~1 TB with `--mem=` (we use 32G) |
| Jupyter / `data*` | 32 GB | n/a |
| GPUs per job | n/a | typically 1–4; more GPUs → more jobs, not one giant job |
| Fair share | n/a | shorter jobs and fewer GPUs get priority; you can run several jobs up to your association limit |

**Do not request a CPU count on this cluster.** `--cpus-per-task` is rejected (`Please do not specify the number of CPUs. There is actually no limit.`). The CS page mentions `-c`; ignore it here.

**Ask only for the memory you need.** Large `--mem` parks the job on fewer fat nodes and delays everyone, including you.

**GPU type (optional):**

```bash
sinfo -o "%25N %50f"          # features per node
sbatch -C a4000 ...           # RTX A4000
sbatch -C ampere ...          # Ampere architecture
sbatch -w rlab2 ...           # pin a node (usually don't)
srun -G a6000:2 nvidia-smi    # two A6000s, CS syntax
```

`snodes` is the local “what does each box look like” command.

---

## 3. One-time setup for this project

```bash
ssh so504@ilab1.cs.rutgers.edu
cd /common/users/so504/Desktop/BehaviorRes
source .venv/bin/activate
```

The venv was created with `--system-site-packages` so it uses **cluster Torch** (`2.7.0+cu126`), not a pip CUDA download (that hit quota). Confirm:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
ls ModelDataRightContra/sessions | wc -l
python Modelv1.py --list-jobs | head
```

You want `60 jobs` for Modelv1, `60` for Modelv2, `150` for Baselinev1.

Sync updated code from your laptop (exclude the venv):

```bash
rsync -av --exclude '.venv' --exclude 'slurm/logs' --exclude '__pycache__' \
  /Users/Stefan/Desktop/BrainMap/BehaviorRes/ \
  so504@ilab1:/common/users/so504/Desktop/BehaviorRes/
```

Scripts must be **Unix LF**. If bash says `$'\r': command not found` or `set: pipefail`:

```bash
sed -i 's/\r$//' slurm/submit.sh slurm/train.sbatch
```

---

## 4. How this repo’s jobs are structured

`bash slurm/submit.sh Modelv1.py` asks Python for the config grid, then submits **one Slurm array**. Each array task is one `(behavior, holdout, pool, ablation)` and trains a **1-d** model.

| Script | Array size | Writes under |
|---|---|---|
| `Modelv1.py` | 60 | `Modelv1/<behavior>/jobs/<index>_<tag>/` |
| `Modelv2.py` | 60 | `Modelv2/<behavior>/jobs/...` |
| `Baselinev1.py` | 150 | `Baselinev1/<behavior>/jobs/...` |

Behaviors: `wheel_speed`, `paw_vx`, `paw_vy`, `paw_vz`, `paw_speed`.

Each job dir gets `status.json` (`starting` → `training` → `done`), `epochs.log`, `scores.csv`.

Stdout/stderr: `slurm/logs/<jobname>_<arrayid>_<task>.out` and `.err`.

`train.sbatch` uses `BEH_ROOT` / `SLURM_SUBMIT_DIR`. **Do not** derive the project root from `BASH_SOURCE` — Slurm copies the script into `/var/spool/slurmd/` and `mkdir` will fail with `Permission denied`.

Worker resources in `train.sbatch`: `--mem=32G`, `--time=12:00:00` (under the 7-day kill limit). GPU is requested by `submit.sh` as `--gpus=1` (same idea as CS’s `-G 1`).

---

## 5. Submit training

Always from the project directory, venv on:

```bash
cd /common/users/so504/Desktop/BehaviorRes
source .venv/bin/activate
```

**GPU array (normal):**

```bash
bash slurm/submit.sh Modelv1.py
```

**Same for the others:**

```bash
bash slurm/submit.sh Modelv2.py
bash slurm/submit.sh Baselinev1.py
```

**CPU-only array** (no `--gpus`):

```bash
bash slurm/submit.sh Modelv1.py --cpu
```

**Cap how many run at once** (example: 8 GPUs at a time). Extra `sbatch` flags go after the script name; a later `--array` overrides the default `0-59`:

```bash
bash slurm/submit.sh Modelv1.py --array=0-59%8
```

**Email** (CS documents `--mail-type` / `--mail-user` on both `sbatch` and `srun`):

```bash
bash slurm/submit.sh Modelv1.py \
  --mail-type=BEGIN,END,FAIL,TIME_LIMIT \
  --mail-user=so504@scarletmail.rutgers.edu
```

Useful `--mail-type` values: `BEGIN`, `END`, `FAIL`, `TIME_LIMIT`, `ALL`. Array jobs can generate a lot of mail; `END,FAIL` is usually enough.

**Time / memory overrides:**

```bash
bash slurm/submit.sh Modelv1.py --time=24:00:00 --mem=16G
```

Default memory if you omit `--mem` is **40 GB** on this scheduler.

**One config only** (debug, after you know the index from `--list-jobs`):

```bash
python Modelv1.py --list-jobs
sbatch --job-name=Modelv1-debug --chdir="$PWD" --gpus=1 --mem=32G --time=02:00:00 \
  --output="$PWD/slurm/logs/debug_%j.out" \
  --export=ALL,BEH_SCRIPT=Modelv1.py,BEH_DEVICE=auto,BEH_ROOT="$PWD" \
  slurm/train.sbatch
# then that worker still needs SLURM_ARRAY_TASK_ID. Easier:
python Modelv1.py --job 0 --device cuda
```

The last line is sequential on a GPU **only if** you already have a GPU via `srun`:

```bash
srun -G 1 --mem=32G --time=02:00:00 --pty bash
source .venv/bin/activate
python Modelv1.py --job 0 --device cuda
```

**Local / login-node sequential CPU** (slow, hits interactive limits; don’t do long training this way):

```bash
python Modelv1.py --device cpu
```

---

## 6. While jobs are queued or running

```bash
squeue -u $USER
squeue -j 367885
watch -n 30 squeue -u $USER
```

Columns: `ST` is `PD` pending, `R` running, `CG` completing. `NODELIST(REASON)` for pending jobs:

| Reason | Meaning |
|---|---|
| `AssocGrpGRES` | you hit your account’s GPU association limit; extras wait. Normal. |
| `Resources` | no free GPU/memory matching the request |
| `Priority` | others are ahead in fair-share |
| `ReqNodeNotAvail` | you pinned a busy node |

Your 60-task Modelv1 array often runs ~12 at a time (`0–11` running, `[12-59]` pending `AssocGrpGRES`). That is the scheduler giving a fair share, not a failure.

```bash
scontrol show job 367885_0          # one task
snodes                              # node inventory
sinfo -Nel                          # nodes, memory, features
srun -G 1 nvidia-smi                # “is any GPU free?” (waits if not)
```

Cancel:

```bash
scancel 367885                      # whole array
scancel 367885_3                    # one task
scancel -u $USER                    # everything of yours (careful)
```

---

## 7. Logs and training progress

Replace `367885` with the id `sbatch` printed.

```bash
ls slurm/logs/Modelv1_367885_*.out | wc -l
cat slurm/logs/Modelv1_367885_0.out
cat slurm/logs/Modelv1_367885_0.err
tail -n 50 slurm/logs/Modelv1_367885_0.out
```

`tail -f` **hangs with no new lines** when the task is not printing (finished, failed, or still starting). Ctrl-C. Empty `.out` plus a 2-second `FAILED` usually means the batch script died before Python.

Healthy first lines:

```text
host=ilab1  cwd=/common/users/so504/Desktop/BehaviorRes  cuda=0  script=Modelv1.py  job=0  device=auto
torch 2.7.0+cu126 cuda True
```

Per-config status (this is the progress file, updated each epoch):

```bash
find Modelv1 -name status.json | wc -l
find Modelv1 -name status.json | head
cat Modelv1/wheel_speed/jobs/*/status.json
tail Modelv1/wheel_speed/jobs/*/epochs.log
```

Quick counts:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from collections import Counter
c = Counter()
for p in Path("Modelv1").glob("*/jobs/*/status.json"):
    c[json.loads(p.read_text()).get("state", "?")] += 1
print(dict(c) or "no status.json yet")
PY
```

Accounting after the fact:

```bash
sacct -j 367885 --format=JobID,State,Elapsed,ExitCode,MaxRSS,NodeList -P | head -80
sacct -j 367885 --format=JobID,State,Elapsed,ExitCode -P | grep -v '\.batch\|\.extern' | cut -d'|' -f2 | sort | uniq -c
```

`ExitCode` `0:0` = success, `1:0` = script error. Ignore `.extern` (always COMPLETED). `.batch` is the actual bash wrapper.

History for ~6 months (CS example):

```bash
sacct -u $USER -S now-180days \
  -o JobID,User,MaxRSS,MaxVMSize,ReqMem,Submit,Start,State,AllocTRES,Nodelist,Reason
```

---

## 8. When the array is done

`squeue -u $USER` empty for that job, and `sacct` shows `COMPLETED` for `_0` … `_59`.

Merge per-task CSVs into each behavior folder:

```bash
python Modelv1.py --aggregate
python Modelv2.py --aggregate
python Baselinev1.py --aggregate
```

Rebuild Baseline plots from cached scores:

```bash
python Baselinev1.py --plots-only
```

Outputs live in `Modelv1/<behavior>/scores.csv`, `REPORT.md`, plots, `jobs/`.

---

## 9. Failures you already hit (and the fix)

| Symptom | Cause | Fix |
|---|---|---|
| `$'\r': command not found` / `set: pipefail` | Windows CRLF in `.sh` | `sed -i 's/\r$//' slurm/*.sh slurm/*.sbatch` |
| `Please do not specify the number of CPUs` | `#SBATCH --cpus-per-task` | do not set CPU count |
| `mkdir ... /var/spool/slurmd/slurm: Permission denied`, FAILED in 3s, ~2 MB RSS | `ROOT` from `BASH_SOURCE` | use `BEH_ROOT` / `SLURM_SUBMIT_DIR` (current `train.sbatch`) |
| `AssocGrpGRES` pending | GPU association / fair share | wait; or `--array=0-59%4` to request fewer at once |
| `couldn't chdir to /common/home/<netid>: Permission denied` | stale SSH host key for the Slurm hop | `ssh-keygen -R '[slurm.cs.rutgers.edu]:23'` then `srun -G 1 nvidia-smi` and type `yes` ([CS troubleshooting](https://resources.cs.rutgers.edu/docs/scheduler-for-gpu-jobs/)) |
| Torch pip downloading cuDNN, quota | installing CUDA Torch into the venv | cluster Torch + `--system-site-packages`; do not `pip install torch` |
| `nvidia-smi` empty on login node | no GPU without Slurm | `srun -G 1 nvidia-smi` or look inside a job log |

---

## 10. Interactive limits if you are *not* using Slurm

From the [limitations page](https://resources.cs.rutgers.edu/docs/limitation-enforced-on-cs-linux-machines/):

```bash
keep-job 30          # keep a disconnected session ~30 daytime hours
sessions             # CPU time and memory for your sessions
sessions -l
```

CPU hours are **usage**, not wall clock. Renew `keep-job` before it expires or the system kills the session. This does **not** apply to `sbatch` jobs.

Home directory `/common/home` has a quota. Project files on `/common/users/...` follow CS storage policy; see their storage page if you fill the disk.

`keep-job` / iLab 80 GB / no-GPU-without-Slurm are why training belongs in `sbatch`, not `tmux` on a login node.

---

## 11. Common Slurm commands (CS list, trimmed)

| Command | Use |
|---|---|
| `sbatch` | submit batch script / our `submit.sh` |
| `srun` | interactive or one-shot on allocated GPUs |
| `squeue` | queue |
| `scancel` | kill job / array |
| `sinfo` / `sinfo -Nel` | partitions and nodes |
| `snodes` | local full node dump |
| `sacct` | finished-job accounting |
| `scontrol show job ID` | live job record |
| `sstat` | stats for a **running** step |
| `sprio` | why a pending job has that priority |
| `sshare` | fair-share usage |

CS’s recommended interactive test:

```bash
srun -G 1 --pty bash -l
```

Graphics (RDP / weblogin session only): `srun --x11=first -G 1 --pty ...`

They recommend `#!/bin/bash -l` so `~/.bash_profile` is read. Our worker uses `#!/bin/bash` plus an explicit `source .venv/bin/activate`.

Do **not** wrap every line in `srun` inside an `sbatch` file on this cluster; CS says that is unnecessary here.

---

## 12. What a full Modelv1 GPU run looks like

```bash
ssh so504@ilab1.cs.rutgers.edu
cd /common/users/so504/Desktop/BehaviorRes
source .venv/bin/activate
python Modelv1.py --list-jobs | head -3
bash slurm/submit.sh Modelv1.py \
  --mail-type=END,FAIL \
  --mail-user=so504@scarletmail.rutgers.edu
# note the Submitted batch job NNNNNN
squeue -u $USER
cat slurm/logs/Modelv1_NNNNNN_0.out
# ... hours later ...
sacct -j NNNNNN --format=JobID,State,Elapsed,ExitCode -P | grep -v '\.batch\|\.extern' | cut -d'|' -f2 | sort | uniq -c
python Modelv1.py --aggregate
```

Copy this file when you `rsync` the `slurm/` directory so the cluster copy stays in sync.
