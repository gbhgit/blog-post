## 🧩 Pre-commit Installation and Setup Guide

### ✅ Step 1: Install Python and pip (if not already installed)

```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

---

### ✅ Step 2: Install `pre-commit` globally (user space)

```bash
pip3 install --user pre-commit
```

> This installs `pre-commit` to `~/.local/bin/`

---

### ✅ Step 3: Add `~/.local/bin` to your PATH (if needed)

If `pre-commit` is not found after installing, add this to your shell config:

#### For Bash:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
---

### ✅ Step 4: Create a `.pre-commit-config.yaml` in your repo root

Example:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v2.3.0
    hooks:
    -   id: check-yaml
    -   id: end-of-file-fixer
    -   id: trailing-whitespace
-   repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
    -   id: black
```

---

### ✅ Step 5: Install the Git hook

Inside your repository:

```bash
pre-commit install
```

---

### ✅ Notes

- `pre-commit` will now run automatically on `git commit`.
- You can add more hooks from [https://pre-commit.com/hooks.html](https://pre-commit.com/hooks.html)
