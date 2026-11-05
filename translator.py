"""Spectral-to-molecular translation from paired Raman and MALDI measurements.

The translator fits a multi-output ridge regression model using Raman spectral
intensities as predictors and MALDI molecular abundances as targets. Inputs are
preprocessed, spatially paired HDF5 datasets; raw-data processing and spatial
registration are outside the scope of this script.

Run:
    python translator.py --train path/to/train.h5 --test path/to/test.h5

Required HDF5 datasets:
    raman_matrix_median: (n_pixels, n_raman_features)
    maldi_matrix:        (n_pixels, n_maldi_features)

Optional test HDF5 datasets for spatial SSIM:
    x_coords, y_coords:  (n_pixels,)

Feature identities and their ordering must match between training and test
files. No pretrained model is required or supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import rankdata
from skimage.metrics import structural_similarity
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


MALDI_KEY = "maldi_matrix"
RAMAN_KEY = "raman_matrix_median"
SSIM_WINDOW_SIZE = 7


def load_paired_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load paired MALDI/Raman feature matrices and optional pixel coordinates."""
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
    """Pearson correlation per MALDI feature (axis=0) or pixel (axis=1).

    Constant vectors have undefined correlation and are returned as NaN.
    """
    a = actual - np.mean(actual, axis=axis, keepdims=True)
    b = predicted - np.mean(predicted, axis=axis, keepdims=True)
    numer = np.sum(a * b, axis=axis)
    denom = np.sqrt(np.sum(a * a, axis=axis) * np.sum(b * b, axis=axis))
    result = np.full(numer.shape, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=result, where=denom > 0)
    return np.clip(result, -1.0, 1.0)


def spearman_by_axis(actual: np.ndarray, predicted: np.ndarray, axis: int) -> np.ndarray:
    """Spearman correlation per feature or pixel, using average ranks for ties.

    Correlation is Pearson's r applied to ranks. Constant vectors are NaN.
    """
    ranked_actual = rankdata(actual, axis=axis, method="average")
    ranked_predicted = rankdata(predicted, axis=axis, method="average")
    return pearson_by_axis(ranked_actual, ranked_predicted, axis=axis)


def featurewise_ssim(
    actual: np.ndarray,
    predicted: np.ndarray,
    x_coords: np.ndarray | None,
    y_coords: np.ndarray | None,
) -> np.ndarray:
    """Compute strictly masked spatial SSIM for each MALDI molecular feature.

    Only 7 x 7 local windows containing measured pixels at every location
    contribute to the SSIM mean. Missing pixels are excluded from both local
    statistics and aggregation. Values at missing locations are merely
    placeholders and do not appear inside any eligible window.

    Features with no eligible windows or no positive intensity range are NaN.
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

    if width < SSIM_WINDOW_SIZE or height < SSIM_WINDOW_SIZE:
        print("SSIM skipped: the test image is smaller than the 7 x 7 SSIM window.")
        return values

    locations = np.ravel_multi_index((y, x), (height, width))
    if np.unique(locations).size != locations.size:
        raise ValueError("Duplicate (x_coords, y_coords) pairs prevent spatial image reconstruction.")

    measured_mask = np.zeros((height, width), dtype=bool)
    measured_mask[y, x] = True
    eligible_centers = binary_erosion(
        measured_mask,
        structure=np.ones((SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE), dtype=bool),
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

        # Only complete measured windows are retained. Their local SSIM values
        # cannot depend on these placeholders at unmeasured locations.
        img_actual.fill(float(actual_values.mean()))
        img_predicted.fill(float(predicted_values.mean()))
        img_actual[y, x] = actual_values
        img_predicted[y, x] = predicted_values

        _, local_ssim = structural_similarity(
            img_actual,
            img_predicted,
            data_range=float(data_range),
            win_size=SSIM_WINDOW_SIZE,
            full=True,
        )
        values[feature_index] = float(np.mean(local_ssim[eligible_centers]))

    return values


def finite_mean(values: np.ndarray) -> float | None:
    """Return the mean of defined values, or None for an all-NaN vector."""
    finite_values = np.asarray(values)[np.isfinite(values)]
    return float(finite_values.mean()) if finite_values.size else None


def save_feature_table(
    path: Path, pearson: np.ndarray, spearman: np.ndarray, ssim: np.ndarray
) -> None:
    """Write one row of evaluation metrics for every MALDI feature index."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["maldi_feature_index", "feature_pearson", "feature_spearman", "feature_ssim"])
        for index, (p, s, structural) in enumerate(zip(pearson, spearman, ssim)):
            writer.writerow([
                index,
                float(p) if np.isfinite(p) else "",
                float(s) if np.isfinite(s) else "",
                float(structural) if np.isfinite(structural) else "",
            ])


def run_translator(train_file: Path, test_file: Path, output_dir: Path, alpha: float, skip_ssim: bool) -> dict:
    """Fit Raman-to-MALDI ridge regression, evaluate, and save results."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("--alpha must be a finite, positive value.")
    if train_file.resolve() == test_file.resolve():
        raise ValueError("Training and test files must be different for held-out evaluation.")

    maldi_train, raman_train, _, _ = load_paired_data(train_file)
    maldi_test, raman_test, x_test, y_test = load_paired_data(test_file)

    if raman_train.shape[1] != raman_test.shape[1]:
        raise ValueError("Training and test Raman feature counts do not match.")
    if maldi_train.shape[1] != maldi_test.shape[1]:
        raise ValueError("Training and test MALDI feature counts do not match.")
    # Feature identities and ordering must also match; validate upstream.

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(raman_train, maldi_train)
    prediction = model.predict(raman_test)

    pixel_pearson = pearson_by_axis(maldi_test, prediction, axis=1)
    pixel_spearman = spearman_by_axis(maldi_test, prediction, axis=1)
    feature_pearson = pearson_by_axis(maldi_test, prediction, axis=0)
    feature_spearman = spearman_by_axis(maldi_test, prediction, axis=0)
    feature_ssim = (
        np.full(maldi_test.shape[1], np.nan, dtype=np.float64)
        if skip_ssim
        else featurewise_ssim(maldi_test, prediction, x_test, y_test)
    )

    summary = {
        "method": "Raman-to-MALDI ridge regression",
        "alpha": float(alpha),
        "n_train_pixels": int(maldi_train.shape[0]),
        "n_test_pixels": int(maldi_test.shape[0]),
        "n_raman_features": int(raman_train.shape[1]),
        "n_maldi_features": int(maldi_train.shape[1]),
        "pixel_pearson_mean": finite_mean(pixel_pearson),
        "pixel_spearman_mean": finite_mean(pixel_spearman),
        "feature_pearson_mean": finite_mean(feature_pearson),
        "feature_spearman_mean": finite_mean(feature_spearman),
        "feature_ssim_mean": finite_mean(feature_ssim),
        "r2_uniform_average": float(r2_score(maldi_test, prediction, multioutput="uniform_average")),
        "ssim_note": (
            "Strictly masked SSIM: mean of 7 x 7 local SSIM scores for centers "
            "with fully measured neighborhoods; missing pixels are excluded"
            if not skip_ssim and x_test is not None
            else "SSIM not calculated"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "translator_predictions.npy", prediction)
    np.savez_compressed(
        output_dir / "translator_coefficients.npz", coef=model.coef_, intercept=model.intercept_
    )
    np.save(output_dir / "translator_pixel_pearson.npy", pixel_pearson)
    np.save(output_dir / "translator_pixel_spearman.npy", pixel_spearman)
    save_feature_table(
        output_dir / "translator_feature_metrics.csv",
        feature_pearson, feature_spearman, feature_ssim,
    )
    (output_dir / "translator_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral-to-molecular translation with ridge regression.")
    parser.add_argument("--train", type=Path, required=True, help="Training HDF5 with paired Raman and MALDI matrices.")
    parser.add_argument("--test", type=Path, required=True, help="Independent test HDF5 with paired Raman and MALDI matrices.")
    parser.add_argument("--output-dir", type=Path, default=Path("translator_results"), help="Directory for predictions and metrics.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Ridge regularization strength (default: 0.01).")
    parser.add_argument("--skip-ssim", action="store_true", help="Skip spatial SSIM calculation.")
    args = parser.parse_args()

    summary = run_translator(args.train, args.test, args.output_dir, args.alpha, args.skip_ssim)
    print(json.dumps(summary, indent=2))
    print(f"Results saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
