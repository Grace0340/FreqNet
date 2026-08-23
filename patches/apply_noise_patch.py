"""Idempotently patch TSLib's exp_long_term_forecasting.py test() for the
noise-robustness study (P2):

  * env TSLIB_NOISE_SIGMA=<s>: adds Gaussian noise s * per-channel-std to the
    test inputs;
  * noisy outputs are isolated: results/test_results folders get a
    "_noise<s>" suffix so clean metrics/perwindow files are never overwritten;
  * env TSLIB_RESULT_FILE: redirects the appended text summary (default
    result_long_term_forecast.txt) so noisy runs do not pollute the main file.

Usage (on the server):
  python apply_noise_patch.py /root/autodl-tmp/Time-Series-Library
"""
import re
import sys
from pathlib import Path

MARKER = "TSLIB_NOISE_SIGMA"


def indent_of(src, idx):
    line_start = src.rfind("\n", 0, idx) + 1
    m = re.match(r"[ \t]*", src[line_start:])
    return src[line_start:line_start + m.end()]


def main(ts_root):
    target = Path(ts_root) / "exp" / "exp_long_term_forecasting.py"
    src = target.read_text(encoding="utf-8")
    if MARKER in src:
        print("already patched, nothing to do")
        return

    # work only inside test()
    tdef = src.index("def test(self")
    tend = src.find("\n    def ", tdef)
    if tend == -1:
        tend = len(src)
    region = src[tdef:tend]

    # 1) read sigma once at the top of test()
    header_end = region.index("\n", region.index(":")) + 1
    sigma_line = (
        "        _noise_sig = float(__import__('os').environ.get('TSLIB_NOISE_SIGMA', '0') or 0)\n"
    )
    region = region[:header_end] + sigma_line + region[header_end:]

    # 2) inject noise after the batch_x device transfer
    anchor = "batch_x = batch_x.float().to(self.device)"
    i = region.index(anchor)
    ind = indent_of(region, i)
    insert = (
        f"\n{ind}if _noise_sig > 0:"
        f"\n{ind}    batch_x = batch_x + _noise_sig * batch_x.std(dim=1, keepdim=True) * torch.randn_like(batch_x)"
    )
    j = i + len(anchor)
    region = region[:j] + insert + region[j:]

    # 3) suffix all output folders when noisy
    def add_suffix(reg, pattern):
        out, pos = [], 0
        for m in re.finditer(pattern, reg):
            ind2 = indent_of(reg, m.start())
            j2 = m.end()
            ins = (
                f"\n{ind2}if _noise_sig > 0:"
                f"\n{ind2}    folder_path = folder_path[:-1] + '_noise' + str(_noise_sig) + '/'"
            )
            out.append(reg[pos:j2] + ins)
            pos = j2
        out.append(reg[pos:])
        return "".join(out)

    region = add_suffix(region, r"folder_path = '\./(?:test_)?results/' \+ setting \+ '/'")

    # 4) redirect the text summary file
    region = region.replace(
        '"result_long_term_forecast.txt"',
        "__import__('os').environ.get('TSLIB_RESULT_FILE', 'result_long_term_forecast.txt')",
    ).replace(
        "'result_long_term_forecast.txt'",
        "__import__('os').environ.get('TSLIB_RESULT_FILE', 'result_long_term_forecast.txt')",
        1,
    )

    patched = src[:tdef] + region + src[tend:]
    backup = target.with_suffix(".py.prenoise")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"patched {target} (backup at {backup})")
    import py_compile
    py_compile.compile(str(target), doraise=True)
    print("syntax check OK")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
