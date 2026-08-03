# Getting a Zenodo DOI (required by Scientific Reports)

GitHub alone is **not** enough for the editor: you need a permanent DOI
(e.g. Zenodo). The repository and `v1.0.0` release are already on GitHub:

- Repository: https://github.com/Rezaul228/cxr_dual_branch_retrieval
- Release: https://github.com/Rezaul228/cxr_dual_branch_retrieval/releases/tag/v1.0.0

## Recommended: GitHub → Zenodo archive

1. Sign in at https://zenodo.org (use ORCID or GitHub login).
2. Go to **GitHub** under your Zenodo account settings:
   https://zenodo.org/account/settings/github/
3. Enable the repository **Rezaul228/cxr_dual_branch_retrieval**.
4. Create a **new** GitHub release after enabling (Zenodo archives on release events).
   Example (from this machine):

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/abedin/Developments/cxr_dual_branch_retrieval
git tag -a v1.0.1 -m "Zenodo archive trigger for Scientific Reports"
git push origin v1.0.1
gh release create v1.0.1 --title "v1.0.1 — Zenodo archive" \
  --notes "Archive trigger for Zenodo DOI after enabling GitHub integration."
```

   If `v1.0.0` was created **before** enabling Zenodo, either flip the toggle and
   use **v1.0.1**, or on Zenodo use “Synchronize now” / re-archive if available.

5. Open the resulting Zenodo record and copy the DOI
   (form: `10.5281/zenodo.xxxxxxx`).
6. Put the DOI into:
   - manuscript **Code Availability**
   - response letter
   - this repo `README.md` (replace `XXXXXXX`)

## Alternative: manual Zenodo upload

1. Download the `v1.0.0` source zip from GitHub Releases (or `git archive`).
2. New upload on Zenodo → upload zip → fill metadata (title, authors, license MIT).
3. Publish → copy DOI.

## Code Availability text (fill DOI)

> The source code, preprocessing scripts, evaluation utilities, and released model
> weights supporting this study are available at
> https://github.com/Rezaul228/cxr_dual_branch_retrieval
> (version v1.0.0) and archived at Zenodo
> (DOI: https://doi.org/10.5281/zenodo.XXXXXXX).
> The repository is released under the MIT License. MIMIC-CXR images and reports
> are not redistributed; authorized users must obtain them from PhysioNet.
