# BPT Classification Pipeline

Emission-line fitting and BPT diagram classification using SDSS DR18 spectra.

## What it does
- Queries SDSS for galaxy spectra near a sky position
- Fits Gaussian + linear continuum models to key emission lines
  (Hβ, [O III], Hα, [N II], [S II], [O I])
- Classifies galaxies as star-forming, composite, Seyfert, or LINER
  using the Kewley/Kauffmann BPT demarcation curves
- Produces diagnostic plots of each line fit and the BPT diagram

## Setup
\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Usage
\`\`\`bash
python fit_emission_lines.py --position "0h8m05.6s +14d50m23s" --radius 2.0 --plot
python visualize_fitting.py --position "0h8m05.6s +14d50m23s" --max-spectra 3
\`\`\`
