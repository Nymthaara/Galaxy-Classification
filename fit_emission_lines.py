#!/usr/bin/env python3
"""Download SDSS galaxy spectra and fit common emission lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import coordinates as coords
from astropy.io import fits
from astropy.table import Table, vstack
from astroquery.sdss import SDSS
from lmfit.models import GaussianModel, LinearModel

# Rest-frame vacuum wavelengths in Angstrom
EMISSION_LINES = {
    "OII_3727": 3727.09,
    "NeIII_3869": 3869.86,
    "Hdelta": 4101.74,
    "Hgamma": 4340.46,
    "Hbeta": 4861.33,
    "OIII_4959": 4958.91,
    "OIII_5007": 5006.84,
    "HeI_5876": 5875.62,
    "OI_6300": 6300.31,
    "Halpha": 6562.82,
    "NII_6548": 6548.05,
    "NII_6583": 6583.46,
    "SII_6716": 6716.44,
    "SII_6731": 6730.82,
}


def fit_line(wave, flux, ivar, center, window=40.0):
    """Fit a single emission line with a Gaussian plus linear continuum."""
    mask = np.abs(wave - center) < window
    good = mask & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    w, f = wave[good], flux[good]
    if len(w) < 10:
        return np.nan, np.nan, None

    weights = np.sqrt(ivar[good])
    model = GaussianModel(prefix="g_") + LinearModel(prefix="c_")
    pars = model.make_params(
        g_center=center,
        g_sigma=2.0,
        g_amplitude=np.max(f),
        c_slope=0,
        c_intercept=np.median(f),
    )
    out = model.fit(f, pars, x=w, weights=weights)
    flux_line = out.params["g_amplitude"].value
    flux_err = out.params["g_amplitude"].stderr
    return flux_line, flux_err, out


def extract_spectrum(hdu: fits.HDUList) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed wavelength (A), flux, and inverse variance from an SDSS HDU."""
    data = hdu[1].data
    flux = np.asarray(data["flux"], dtype=float)
    wave = 10 ** np.asarray(data["loglam"], dtype=float)
    ivar = np.asarray(data["ivar"], dtype=float)
    return wave, flux, ivar


def observed_line_centers(redshift: float, lines: dict[str, float] | None = None) -> dict[str, float]:
    """Convert rest-frame line wavelengths to observed frame."""
    lines = lines or EMISSION_LINES
    return {name: rest * (1.0 + redshift) for name, rest in lines.items()}


def fit_all_lines(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    redshift: float,
    window: float = 40.0,
    lines: dict[str, float] | None = None,
) -> Table:
    """Fit all requested emission lines for one spectrum."""
    lines = lines or EMISSION_LINES
    centers = observed_line_centers(redshift, lines)
    rows = []

    for name, center in centers.items():
        flux_line, flux_err, fit_result = fit_line(wave, flux, ivar, center, window=window)
        row = {
            "line": name,
            "rest_wave": lines[name],
            "obs_wave": center,
            "flux": flux_line,
            "flux_err": flux_err,
            "fit_ok": fit_result is not None and fit_result.success,
        }
        if fit_result is not None:
            row["center"] = fit_result.params["g_center"].value
            row["center_err"] = fit_result.params["g_center"].stderr
            row["sigma"] = fit_result.params["g_sigma"].value
            row["sigma_err"] = fit_result.params["g_sigma"].stderr
        else:
            row["center"] = np.nan
            row["center_err"] = np.nan
            row["sigma"] = np.nan
            row["sigma_err"] = np.nan
        rows.append(row)

    return Table(rows=rows)


def plot_line_fit(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    center: float,
    fit_result,
    line_name: str,
    output_path: Path,
    window: float = 40.0,
) -> None:
    """Save a diagnostic plot for one line fit."""
    mask = np.abs(wave - center) < window
    good = mask & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    w, f = wave[good], flux[good]
    err = 1.0 / np.sqrt(ivar[good])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(w, f, yerr=err, fmt=".", color="0.4", ms=3, alpha=0.8, label="data")
    if fit_result is not None:
        x_model = np.linspace(w.min(), w.max(), 300)
        ax.plot(x_model, fit_result.eval(x=x_model), "r-", lw=1.5, label="model")
    ax.axvline(center, color="k", ls="--", alpha=0.4, label="catalog center")
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel("Flux")
    ax.set_title(line_name)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def query_and_download(
    position: str,
    radius_arcmin: float = 2.0,
    data_release: int | None = None,
) -> tuple[Table, list[fits.HDUList]]:
    """Query SDSS for spectra near a sky position and download FITS files."""
    pos = coords.SkyCoord(position, frame="icrs")
    kwargs = {"radius": radius_arcmin * u.arcmin, "spectro": True}
    if data_release is not None:
        kwargs["data_release"] = data_release

    matches = SDSS.query_region(pos, **kwargs)
    if matches is None or len(matches) == 0:
        raise RuntimeError(f"No spectroscopic matches found within {radius_arcmin} arcmin of {position}")

    get_kwargs = {"matches": matches}
    if data_release is not None:
        get_kwargs["data_release"] = data_release
    spectra = SDSS.get_spectra(**get_kwargs)
    return matches, spectra


def process_spectrum(
    match_row,
    hdu: fits.HDUList,
    output_dir: Path,
    window: float,
    make_plots: bool,
) -> Table:
    """Fit emission lines for one SDSS spectrum and optionally save plots."""
    wave, flux, ivar = extract_spectrum(hdu)
    redshift = float(match_row["z"])
    specobjid = int(match_row["specobjid"])

    results = fit_all_lines(wave, flux, ivar, redshift, window=window)
    results["specobjid"] = specobjid
    results["plate"] = int(match_row["plate"])
    results["mjd"] = int(match_row["mjd"])
    results["fiberID"] = int(match_row["fiberID"])
    results["ra"] = float(match_row["ra"])
    results["dec"] = float(match_row["dec"])
    results["z"] = redshift

    if make_plots:
        plot_dir = output_dir / f"spec_{specobjid}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        centers = observed_line_centers(redshift)
        for row in results:
            if not row["fit_ok"]:
                continue
            _, _, fit_result = fit_line(wave, flux, ivar, row["obs_wave"], window=window)
            plot_line_fit(
                wave,
                flux,
                ivar,
                row["obs_wave"],
                fit_result,
                row["line"],
                plot_dir / f"{row['line']}.png",
                window=window,
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SDSS spectra and fit emission lines.")
    parser.add_argument(
        "--position",
        default="0h8m05.6s +14d50m23s",
        help="ICRS sky position (default: example field)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Search radius in arcminutes (default: 2)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=40.0,
        help="Half-width of fitting window in Angstrom (default: 40)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for CSV results and plots (default: output)",
    )
    parser.add_argument(
        "--data-release",
        type=int,
        default=None,
        help="SDSS data release (default: astroquery default)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save per-line diagnostic plots",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Querying SDSS within {args.radius} arcmin of {args.position} ...")
    matches, spectra = query_and_download(args.position, args.radius, args.data_release)
    print(f"Found {len(matches)} spectrum(s).")

    all_results = []
    for i, (match_row, hdu) in enumerate(zip(matches, spectra)):
        specobjid = int(match_row["specobjid"])
        print(f"[{i + 1}/{len(matches)}] Fitting specobjid={specobjid}, z={float(match_row['z']):.4f}")
        results = process_spectrum(match_row, hdu, args.output_dir, args.window, args.plot)
        all_results.append(results)

        out_csv = args.output_dir / f"lines_spec_{specobjid}.csv"
        results.write(out_csv, format="ascii.csv", overwrite=True)
        print(f"  Wrote {out_csv}")

    combined = vstack(all_results)
    combined_path = args.output_dir / "lines_all.csv"
    combined.write(combined_path, format="ascii.csv", overwrite=True)
    print(f"Wrote combined results to {combined_path}")

    # Quick summary for BPT-relevant lines
    bpt_lines = ["Halpha", "Hbeta", "OIII_5007", "NII_6583", "SII_6716", "SII_6731"]
    print("\nLine flux summary (Gaussian amplitude):")
    for spec_id in np.unique(combined["specobjid"]):
        sub = combined[combined["specobjid"] == spec_id]
        z = float(sub["z"][0])
        print(f"  specobjid={spec_id}  z={z:.4f}")
        for line in bpt_lines:
            row = sub[sub["line"] == line]
            if len(row) == 0:
                continue
            flux = row["flux"][0]
            err = row["flux_err"][0]
            if np.isfinite(flux):
                err_str = f" +/- {err:.2g}" if np.isfinite(err) else ""
                print(f"    {line:10s}  {flux:10.2f}{err_str}")
            else:
                print(f"    {line:10s}  fit failed")


if __name__ == "__main__":
    main()
