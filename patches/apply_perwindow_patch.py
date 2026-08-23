"""Idempotently patch TSLib's exp_long_term_forecasting.py so that test()
additionally saves per-window (per test sample) MSE/MAE arrays, which the
paired significance tests need.

Adds right before the metrics.npy save:
    perwindow_mse.npy  -- shape (n_test_windows,)
    perwindow_mae.npy  -- shape (n_test_windows,)

Usage (on the server):
  python apply_perwindow_patch.py /root/autodl-tmp/Time-Series-Library
"""
import sys
from pathlib import Path

MARKER = "perwindow_mse.npy"
ANCHOR = "np.save(folder_path + 'metrics.npy'"
INSERT = (
    "        pw_mse = ((preds - trues) ** 2).mean(axis=(1, 2))\n"
    "        pw_mae = np.abs(preds - trues).mean(axis=(1, 2))\n"
    "        np.save(folder_path + 'perwindow_mse.npy', pw_mse)\n"
    "        np.save(folder_path + 'perwindow_mae.npy', pw_mae)\n"
)


def main(ts_root):
    target = Path(ts_root) / "exp" / "exp_long_term_forecasting.py"
    src = target.read_text(encoding="utf-8")
    if MARKER in src:
        print("already patched, nothing to do")
        return
    if ANCHOR not in src:
        sys.exit(f"anchor not found in {target}; patch manually")
    idx = src.index(ANCHOR)
    line_start = src.rfind("\n", 0, idx) + 1
    patched = src[:line_start] + INSERT + src[line_start:]
    backup = target.with_suffix(".py.prepatch")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"patched {target} (backup at {backup})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
