"""Shared numerical checks. PyTorch is imported only when comparing tensors."""
import json
import os
from pathlib import Path


def numerical_tolerances(dtype):
    return {"rtol": 0.02, "atol": 0.02} if dtype == "bfloat16" else {"rtol": 1e-4, "atol": 1e-5}


class NumericalMismatch(AssertionError):
    """Finite, shape/dtype-compatible outputs failed the numerical tolerance."""

    def __init__(self, report):
        self.report = report
        super().__init__(f"Mismatch in {list(report['failures'])}; max absolute errors: "
                         f"{report['max_abs_errors']}; details in {report['diagnostic_dir']}")


def compare_tensors(actual, expected, dtype, names, exact=False, *, diagnostic_dir=None):
    import torch

    assert len(actual) == len(expected) == len(names)
    errors, failures, diagnostic, statistics = {}, {}, {}, {}
    tolerances = {"rtol": 0, "atol": 0} if exact else numerical_tolerances(dtype)
    for name, observed, reference in zip(names, actual, expected):
        observed = observed.cpu()
        reference = reference.cpu()
        # Structural failures and non-finite values must never become an
        # exploratory numerical warning. The decoder catches only our subclass.
        if observed.shape != reference.shape or observed.dtype != reference.dtype:
            raise ValueError(f"{name}: shape/dtype mismatch: {observed.shape}/{observed.dtype} "
                             f"versus {reference.shape}/{reference.dtype}")
        if not torch.isfinite(observed).all() or not torch.isfinite(reference).all():
            raise ValueError(f"{name}: non-finite values in actual or reference")
        difference = observed.float() - reference.float()
        errors[name] = difference.abs().max().item()
        statistics[name] = {
            "elements": observed.numel(),
            "mismatched_elements": int((~torch.isclose(observed, reference, **tolerances)).sum().item()),
            "mean_abs_error": difference.abs().mean().item(),
            "rms_error": difference.square().mean().sqrt().item(),
        }
        try:
            torch.testing.assert_close(observed, reference, **tolerances)
        except AssertionError as error:
            failures[name] = str(error)
            diagnostic[name] = {"actual": observed, "expected": reference}
    if failures:
        scratch = Path(diagnostic_dir or os.environ["TMPDIR"]).resolve()
        if not scratch.is_relative_to(Path(os.environ["TMPDIR"]).resolve()):
            raise ValueError("Numerical diagnostics must be under TMPDIR")
        scratch.mkdir(parents=True, exist_ok=True)
        report = {"max_abs_errors": errors, "failures": failures, "statistics": statistics,
                  "tolerances": tolerances, "diagnostic_dir": str(scratch)}
        (scratch / "comparison_failure.json").write_text(json.dumps(report, indent=2) + "\n")
        torch.save(diagnostic, scratch / "comparison_failure.pt")
        raise NumericalMismatch(report)
    return errors
