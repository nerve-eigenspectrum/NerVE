# Step-by-Step: Publishing NerVE to GitHub

## Prerequisites
- GitHub account
- Git installed locally
- The nerve-spectral folder from the tar.gz download

---

## Step 1: Create the GitHub repository

1. Go to https://github.com/new
2. Settings:
   - **Owner**: Choose your org `nerve-eigenspectrum` (or your personal account)
   - **Repository name**: `NerVE`
   - **Description**: `NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (ICLR 2026)`
   - **Visibility**: Public
   - **DO NOT** check "Add a README" (we already have one)
   - **DO NOT** check "Add .gitignore" (we already have one)
   - **DO NOT** choose a license (we already have MIT)
3. Click **Create repository**

---

## Step 2: Initialize and push locally

Open terminal, navigate to the nerve-spectral folder:

```bash
cd nerve-spectral

# Initialize git repo
git init

# Add all files
git add .

# Check what will be committed (verify no junk files)
git status

# First commit
git commit -m "Initial release: NerVE eigenspectral framework (ICLR 2026)"

# Set the main branch
git branch -M main

# Add remote (replace with YOUR actual repo URL from Step 1)
git remote add origin https://github.com/nerve-eigenspectrum/NerVE.git

# Push
git push -u origin main
```

---

## Step 3: Update your project page

On your project page (https://nerve-eigenspectrum.github.io/), add the GitHub link:

```html
<a href="https://github.com/nerve-eigenspectrum/NerVE">Code</a>
```

---

## Step 4: Add your Hydra config files

You mentioned you have config files that were not uploaded.
Copy them into the `configs/` directory:

```bash
cp /path/to/your/config.yaml configs/
cp -r /path/to/your/config_overrides/ configs/
git add configs/
git commit -m "Add Hydra configuration files for experiments"
git push
```

---

## Step 5: Update placeholder URLs

There are a few placeholder URLs in the repo that need updating:

1. **README.md**: Replace `https://arxiv.org/abs/XXXX.XXXXX` with your actual arXiv link
2. **README.md**: Replace `https://openreview.net/forum?id=XXXXXXXXXX` in the BibTeX
3. **README.md**: Update the Colab demo link once the notebook is ready
4. **pyproject.toml**: Replace `https://arxiv.org/abs/XXXX.XXXXX` with your actual arXiv link

```bash
# After editing the files:
git add README.md pyproject.toml
git commit -m "Update paper links"
git push
```

---

## Step 6 (Optional): Create a GitHub Release

1. Go to your repo → **Releases** → **Create a new release**
2. Tag: `v0.1.0`
3. Title: `NerVE v0.1.0 — ICLR 2026`
4. Description:
   ```
   Initial release of the NerVE eigenspectral framework.
   
   - Core metrics: SE, PR, EEE, JS
   - Inference-time analyzer for pretrained HuggingFace models
   - Training-time monitoring callback
   - GPT-2 architectural variants for reproducing paper experiments
   ```
5. Click **Publish release**

---

## Step 7 (Optional): Publish to PyPI

Once you've verified everything works:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI (you'll need a PyPI account + API token)
twine upload dist/*
```

After this, anyone can `pip install nerve-spectral`.

---

## Step 8 (Later): Add the demo notebook

Once we build the Colab notebook in Phase 3:

```bash
cp demo_pretrained.ipynb notebooks/
git add notebooks/demo_pretrained.ipynb
git commit -m "Add demo notebook for pretrained model analysis"
git push
```

Then update the Colab link in README.md.

---

## Checklist

- [ ] GitHub repo created
- [ ] Code pushed to main branch
- [ ] Project page updated with repo link
- [ ] Config files added
- [ ] Placeholder URLs replaced (arXiv, OpenReview)
- [ ] GitHub Release created (v0.1.0)
- [ ] (Later) PyPI published
- [ ] (Later) Demo notebook added
