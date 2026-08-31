from cvmeasure.evaluate import _console_safe


def test_console_safe_handles_unicode_with_legacy_windows_encoding():
    report = "U-Net → measure | R² | mm²"

    rendered = _console_safe(report, encoding="cp1252")

    assert rendered == "U-Net -> measure | R^2 | mm^2"
    assert rendered.encode("cp1252")
