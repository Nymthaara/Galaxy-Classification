#!/usr/bin/env python3
"""Visualize SDSS emission-line fits with multi-panel diagnostic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_emission_lines import (
    extract_spectrum,
    fit_line,
    observed_line_centers,
    query_and_download,
)

# Short default list for BPT-style visualization
DEFAULT_LINES = [
    "Hbeta",
    "OIII_5007",
    "OI_6300",
    "Halpha",
    "NII_6583",
    "SII_6716",
    "SII_6731",
]


def parse_line_list(value: str) -> list[str]:
    """Parse comma-separated line names."""
    names = [x.strip() for x in value.split(",") if x.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Line list cannot be empty.")
    return names


def good_window_arrays(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    center: float,
    window: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid arrays in the fitting window."""
    mask = np.abs(wave - center) < window
    good = mask & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    w = wave[good]
    f = flux[good]
    e = 1.0 / np.sqrt(ivar[good])
    return w, f, e


def plot_full_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    redshift: float,
    centers: dict[str, float],
    specobjid: int,
    output_path: Path,
) -> None:
    """Plot full spectrum with expected observed line centers."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(wave, flux, lw=0.8, color="0.2")

    y_min, y_max = np.nanpercentile(flux[np.isfinite(flux)], [2, 98])
    span = y_max - y_min
    ax.set_ylim(y_min - 0.1 * span, y_max + 0.15 * span)

    for name, center in centers.items():
        if wave[0] <= center <= wave[-1]:
            ax.axvline(center, ls="--", lw=0.7, alpha=0.4, color="tab:blue")
            ax.text(center, y_max + 0.03 * span, name, rotation=90, fontsize=7, alpha=0.8, ha="center")

    ax.set_title(f"Full SDSS Spectrum | Galaxy ID={specobjid} | z={redshift:.4f}")
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel("Flux")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_line_panels(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    redshift: float,
    specobjid: int,
    line_names: list[str],
    window: float,
    output_path: Path,
) -> None:
    """Create a two-row panel figure with fit and residual for each line."""
    centers = observed_line_centers(redshift)
    n = len(line_names)
    fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 7.0), sharex="col")

    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, line_name in enumerate(line_names):
        ax_top = axes[0, i]
        ax_bottom = axes[1, i]

        if line_name not in centers:
            ax_top.text(0.5, 0.5, f"Unknown line:\n{line_name}", ha="center", va="center")
            ax_top.set_axis_off()
            ax_bottom.set_axis_off()
            continue

        center = centers[line_name]
        w, f, e = good_window_arrays(wave, flux, ivar, center, window)
        flux_line, flux_err, out = fit_line(wave, flux, ivar, center, window=window)

        if len(w) == 0:
            ax_top.text(0.5, 0.5, "No valid points", ha="center", va="center")
            ax_top.set_title(line_name)
            ax_bottom.set_axis_off()
            continue

        ax_top.errorbar(w, f, yerr=e, fmt=".", ms=3, color="0.35", alpha=0.8)
        ax_top.axvline(center, color="k", ls="--", lw=1.0, alpha=0.5)

        if out is not None:
            x_model = np.linspace(w.min(), w.max(), 400)
            y_model = out.eval(x=x_model)
            y_at_w = out.eval(x=w)
            resid = f - y_at_w

            ax_top.plot(x_model, y_model, color="tab:red", lw=1.6)
            ax_bottom.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.5)
            ax_bottom.errorbar(w, resid, yerr=e, fmt=".", ms=3, color="tab:green", alpha=0.8)
        else:
            ax_bottom.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.5)
            ax_bottom.text(0.5, 0.5, "fit failed", ha="center", va="center")

        err_text = f"{flux_err:.2g}" if flux_err is not None and np.isfinite(flux_err) else "nan"
        ax_top.set_title(f"{line_name}\nA={flux_line:.2f} +/- {err_text}", fontsize=10)
        ax_bottom.set_xlabel("Wavelength (A)")
        ax_top.set_ylabel("Flux")
        ax_bottom.set_ylabel("Residual")

    fig.suptitle(f"Line Fits | Galaxy ID={specobjid} | z={redshift:.4f}", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SDSS emission-line fits.")
    parser.add_argument(
        "--position",
        default="0h8m05.6s +14d50m23s",
        help="ICRS sky position string.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Search radius in arcminutes.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=40.0,
        help="Half-width of fitting window in Angstrom.",
    )
    parser.add_argument(
        "--max-spectra",
        type=int,
        default=3,
        help="Maximum number of matched spectra to visualize.",
    )
    parser.add_argument(
        "--lines",
        type=parse_line_list,
        default=DEFAULT_LINES,
        help="Comma-separated line names (e.g. Hbeta,OIII_5007,Halpha,NII_6583).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/visualizations"),
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--data-release",
        type=int,
        default=None,
        help="SDSS data release (optional).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Querying SDSS around {args.position} (radius={args.radius} arcmin)...")
    matches, spectra = query_and_download(args.position, args.radius, args.data_release)
    n_show = min(len(matches), len(spectra), max(args.max_spectra, 1))
    print(f"Found {len(matches)} spectrum(s); visualizing first {n_show}.")

    for i in range(n_show):
        row = matches[i]
        hdu = spectra[i]
        specobjid = int(row["specobjid"])
        redshift = float(row["z"])
        wave, flux, ivar = extract_spectrum(hdu)
        centers = observed_line_centers(redshift)

        full_path = args.output_dir / f"spec_{specobjid}_full.png"
        panel_path = args.output_dir / f"spec_{specobjid}_line_panels.png"

        plot_full_spectrum(wave, flux, redshift, centers, specobjid, full_path)
        plot_line_panels(
            wave=wave,
            flux=flux,
            ivar=ivar,
            redshift=redshift,
            specobjid=specobjid,
            line_names=args.lines,
            window=args.window,
            output_path=panel_path,
        )

        print(f"[{i + 1}/{n_show}] specobjid={specobjid} z={redshift:.4f}")
        print(f"  wrote {full_path}")
        print(f"  wrote {panel_path}")


if __name__ == "__main__":
    main()
