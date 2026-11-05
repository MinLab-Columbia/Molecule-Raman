"""Molecular-to-spectral decoding from paired MALDI and Raman measurements.

The decoder fits a multi-output ridge regression model using MALDI molecular
abundances as predictors and Raman spectral intensities as targets. Inputs are
preprocessed, spatially paired HDF5 datasets; raw-data processing and spatial
registration are outside the scope of this script.

Run:
    python decoder.py --train path/to/train.h5 --test path/to/test.h5

Required HDF5 datasets:
    maldi_matrix:        (n_pixels, n_maldi_features)
    raman_matrix_median: (n_pixels, n_raman_features)

Optional test HDF5 datasets for spatial SSIM:
    x_coords, y_coords:  (n_pixels,)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


MALDI_KEY = "maldi_matrix"
RAMAN_KEY = "raman_matrix_median"


def load_paired_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load paired 2D feature matrices and optional spatial pixel coordinates."""
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, "r") as handle:
        missing = [key for key in (MALDI_KEY, RAMAN_KEY) if key not in handle]
        if missing:
            raise KeyError(f"Missing dataset(s) in {path}: {', '.join(missing)}")

        maldi = np.asarray(handle[MALDI_KEY][:], dtype=np.float64)
        raman = np.asarray(handle[RAMAN_KEY][:], dtype=np.float64)
        x = np.asarray(handle["x_coords"][:]).reshape(-1) if "x_coords" in handle else None
        y = np.asarray(handle["y_coords"][:]).reshape(-1) if "y_coords" in handle else None

    if maldi.ndim != 2 or raman.ndim != 2:
        raise ValueError(f"Both feature matrices in {path} must be two-dimensional.")
    if maldi.shape[0] != raman.shape[0]:
        raise ValueError(f"Unpaired rows in {path}: MALDI {maldi.shape}, Raman {raman.shape}.")
    if maldi.shape[0] < 2 or maldi.shape[1] == 0 or raman.shape[1] == 0:
        raise ValueError(f"Insufficient samples or features in {path}.")
    if not np.isfinite(maldi).all() or not np.isfinite(raman).all():
        raise ValueError(f"Feature matrices in {path} contain NaN or infinity. Clean them before training.")
    if (x is None) != (y is None):
        raise ValueError(f"Include both x_coords and y_coords, or neither, in {path}.")
    if x is not None and (len(x) != len(maldi) or len(y) != len(maldi)):
        raise ValueError(f"Spatial coordinates in {path} must have one entry per pixel.")

    return maldi, raman, x, y


def pearson_by_axis(actual: np.ndarray, predicted: np.ndarray, axis: int) -> np.ndarray:
    """Pearson correlation along axis 0 (features) or 1 (pixels).

    Constant vectors have undefined correlation and are returned as NaN.
    """
    a = actual - np.mean(actual, axis=axis, keepdims=True)
    b = predicted - np.mean(predicted, axis=axis, keepdims=True)
    numer = np.sum(a * b, axis=axis)
    denom = np.sqrt(np.sum(a * a, axis=axis) * np.sum(b * b, axis=axis))
    result = np.full(numer.shape, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=result, where=denom > 0)
    return np.clip(result, -1.0, 1.0)


def featurewise_ssim(
    actual: np.ndarray,
    predicted: np.ndarray,
    x_coords: np.ndarray | None,
    y_coords: np.ndarray | None,
) -> np.ndarray:
    """Compute strictly masked spatial SSIM for each Raman channel.

    SSIM is averaged only at pixel centers whose entire 7 x 7 neighborhood
    contains measured pixels. Missing locations and partial neighborhoods are
    excluded from both the local statistics and the final average. Temporary
    values at missing image locations are never used in an eligible window.

    Channels without any eligible neighborhoods or a positive data range are
    reported as NaN.
    """
    values = np.full(actual.shape[1], np.nan, dtype=np.float64)
    if x_coords is None or y_coords is None:
        return values

    if not np.isfinite(x_coords).all() or not np.isfinite(y_coords).all():
        raise ValueError("Spatial coordinates contain NaN or infinity.")
    if not np.array_equal(x_coords, np.round(x_coords)) or not np.array_equal(y_coords, np.round(y_coords)):
        raise ValueError("Spatial coordinates must represent integer pixel positions.")

    x = x_coords.astype(np.int64)
    y = y_coords.astype(np.int64)
    x = x - x.min()
    y = y - y.min()
    width, height = int(x.max()) + 1, int(y.max()) + 1

    window_size = 7  # matches skimage's standard uniform-window SSIM
    if width < window_size or height < window_size:
        print("SSIM skipped: the test image is smaller than the 7 x 7 SSIM window.")
        return values

    locations = np.ravel_multi_index((y, x), (height, width))
    if np.unique(locations).size != locations.size:
        raise ValueError("Duplicate (x_coords, y_coords) pairs prevent spatial image reconstruction.")

    measured_mask = np.zeros((height, width), dtype=bool)
    measured_mask[y, x] = True
    eligible_centers = binary_erosion(
        measured_mask,
        structure=np.ones((window_size, window_size), dtype=bool),
        border_value=0,
    )
    if not eligible_centers.any():
        print("SSIM skipped: no complete 7 x 7 neighborhood of measured pixels is available.")
        return values

    img_actual = np.empty((height, width), dtype=np.float64)
    img_predicted = np.empty((height, width), dtype=np.float64)

    for feature_index in range(actual.shape[1]):
        actual_values = actual[:, feature_index]
        predicted_values = predicted[:, feature_index]
        data_range = max(actual_values.max(), predicted_values.max()) - min(
            actual_values.min(), predicted_values.min()
        )
        if data_range <= 0:
            continue

        # skimage expects finite rectangular arrays. Fill unmeasured locations
        # only as a numerical placeholder: eligible windows contain measured
        # pixels exclusively, so these values do not enter the SSIM result.
        img_actual.fill(float(actual_values.mean()))
        img_predicted.fill(float(predicted_values.mean()))
        img_actual[y, x] = actual_values
        img_predicted[y, x] = predicted_values

        _, local_ssim = structural_similarity(
            img_actual,
            img_predicted,
            data_range=float(data_range),
            win_size=window_size,
            full=True,
        )
        values[feature_index] = float(np.mean(local_ssim[eligible_centers]))

    return values


def finite_mean(values: np.ndarray) -> float | None:
    """Return a JSON-compatible mean, or None when values are all undefined."""
    finite_values = np.asarray(values)[np.isfinite(values)]
    return float(finite_values.mean()) if finite_values.size else None


def save_feature_table(path: Path, pearson: np.ndarray, ssim: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raman_feature_index", "feature_pearson", "feature_ssim"])
        for index, (p, s) in enumerate(zip(pearson, ssim)):
            writer.writerow([index, float(p) if np.isfinite(p) else "", float(s) if np.isfinite(s) else ""])


def run_decoder(train_file: Path, test_file: Path, output_dir: Path, alpha: float, skip_ssim: bool) -> dict:
    """Fit the decoder, predict test spectra, and save evaluation results."""
    if alpha <= 0:
        raise ValueError("--alpha must be positive.")
    if train_file.resolve() == test_file.resolve():
        raise ValueError("Training and test files must be different for held-out evaluation.")

    maldi_train, raman_train, _, _ = load_paired_data(train_file)
    maldi_test, raman_test, x_test, y_test = load_paired_data(test_file)

    if maldi_train.shape[1] != maldi_test.shape[1]:
        raise ValueError("Training and test MALDI feature counts do not match.")
    if raman_train.shape[1] != raman_test.shape[1]:
        raise ValueError("Training and test Raman feature counts do not match.")
    # Feature identities and ordering must also match; check these upstream.

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(maldi_train, raman_train)
    prediction = model.predict(maldi_test)

    pixel_pearson = pearson_by_axis(raman_test, prediction, axis=1)
    feature_pearson = pearson_by_axis(raman_test, prediction, axis=0)
    feature_ssim = (
        np.full(raman_test.shape[1], np.nan)
        if skip_ssim
        else featurewise_ssim(raman_test, prediction, x_test, y_test)
    )

    summary = {
        "method": "MALDI-to-Raman ridge regression",
        "alpha": alpha,
        "n_train_pixels": int(maldi_train.shape[0]),
        "n_test_pixels": int(maldi_test.shape[0]),
        "n_maldi_features": int(maldi_train.shape[1]),
        "n_raman_features": int(raman_train.shape[1]),
        "pixel_pearson_mean": finite_mean(pixel_pearson),
        "feature_pearson_mean": finite_mean(feature_pearson),
        "feature_ssim_mean": finite_mean(feature_ssim),
        "r2_uniform_average": float(r2_score(raman_test, prediction, multioutput="uniform_average")),
        "ssim_note": (
            "Strictly masked SSIM: mean of 7 x 7 local SSIM scores for centers "
            "with fully measured neighborhoods; missing pixels are excluded"
            if not skip_ssim and x_test is not None
            else "SSIM not calculated"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "decoder_predictions.npy", prediction)
    np.savez_compressed(output_dir / "decoder_coefficients.npz", coef=model.coef_, intercept=model.intercept_)
    np.save(output_dir / "decoder_pixel_pearson.npy", pixel_pearson)
    save_feature_table(output_dir / "decoder_feature_metrics.csv", feature_pearson, feature_ssim)
    (output_dir / "decoder_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Molecular-to-spectral decoding with ridge regression.")
    parser.add_argument("--train", type=Path, required=True, help="Training HDF5 file with paired MALDI and Raman matrices.")
    parser.add_argument("--test", type=Path, required=True, help="Independent test HDF5 file with paired MALDI and Raman matrices.")
    parser.add_argument("--output-dir", type=Path, default=Path("decoder_results"), help="Directory for model outputs and metrics.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Ridge regularization strength (default: 0.01).")
    parser.add_argument("--skip-ssim", action="store_true", help="Skip spatial SSIM calculation.")
    args = parser.parse_args()

    summary = run_decoder(args.train, args.test, args.output_dir, args.alpha, args.skip_ssim)
    print(json.dumps(summary, indent=2))
    print(f"Results saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
