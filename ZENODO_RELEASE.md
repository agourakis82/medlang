# Zenodo Release Instructions

## Release Information

- **Version**: v0.7.0-migration
- **Release Date**: December 25, 2025
- **Status**: Final Release - Repository Archived

## Steps to Publish on Zenodo

### 1. Create GitHub Release

1. Go to: https://github.com/agourakis82/medlang/releases/new
2. **Tag**: Select `v0.7.0-migration`
3. **Title**: `Release v0.7.0 - MedLang → Sounio Migration Complete`
4. **Description**: Copy from `RELEASE_v0.7.0_MIGRATION.md`
5. Check: **"Set as the latest release"**
6. Click **"Publish release"**

### 2. Connect to Zenodo

1. Go to: https://zenodo.org/account/settings/github/
2. Find `agourakis82/medlang` in the list
3. Toggle **ON** to enable automatic releases
4. Zenodo will automatically create a DOI when you publish the GitHub release

### 3. Verify Zenodo Metadata

After the release is published, Zenodo will:
- Create a DOI (e.g., `10.5281/zenodo.1234567`)
- Use metadata from `.zenodo.json`
- Create a citable record

### 4. Update Citation

Once Zenodo assigns a DOI, update:
- `RELEASE_v0.7.0_MIGRATION.md` with the actual DOI
- Any documentation that references the release

## Metadata File

The `.zenodo.json` file contains:
- Title and description
- Author information
- Keywords
- License
- Related identifiers (Sounio, Demetrios)
- Version and publication date

## Citation Format (After DOI Assignment)

```bibtex
@software{medlang2025,
  title = {MedLang: A Medical-Native Programming Language for Computational Pharmacology},
  author = {Agourakis, Demetrios Chiuratto},
  year = {2025},
  version = {0.7.0},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://doi.org/10.5281/zenodo.XXXXXXX},
  note = {Unified into Sounio Programming Language. See https://github.com/sounio-lang/sounio}
}
```

## Archive Notice

After publishing on Zenodo:

1. **Archive Repository**:
   - Go to: https://github.com/agourakis82/medlang/settings
   - Scroll to "Danger Zone"
   - Click "Archive this repository"
   - Confirm

2. **Update Zenodo Record** (if needed):
   - Add note about repository being archived
   - Link to Sounio as continuation

---

**Note**: The Zenodo DOI will be assigned automatically when the GitHub release is published and Zenodo is connected.

