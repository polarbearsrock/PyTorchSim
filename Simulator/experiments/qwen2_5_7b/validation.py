"""Shared numerical checks. PyTorch is imported only when comparing tensors."""
import json
import os
from pathlib import Path


def numerical_tolerances(dtype):
    return {"rtol": 0.02, "atol": 0.02} if dtype == "bfloat16" else {"rtol": 1e-4, "atol": 1e-5}


def compare_tensors(actual, expected, dtype, names, exact=False):
    import torch

    assert len(actual) == len(expected) == len(names)
    errors, failures, diagnostic = {}, {}, {}
    for name, observed, reference in zip(names, actual, expected):
        observed = observed.cpu()
        errors[name] = (observed.float() - reference.float()).abs().max().item()
        try:
            torch.testing.assert_close(observed, reference, **({"rtol": 0, "atol": 0} if exact else numerical_tolerances(dtype)))
        except AssertionError as error:
            failures[name] = str(error)
            diagnostic[name] = {"actual": observed, "expected": reference}
    if failures:
        scratch = Path(os.environ["TMPDIR"])
        (scratch / "comparison_failure.json").write_text(json.dumps({"max_abs_errors": errors, "failures": failures}, indent=2) + "\n")
        torch.save(diagnostic, scratch / "comparison_failure.pt")
        raise AssertionError(f"Mismatch in {list(failures)}; max absolute errors: {errors}; details in comparison_failure.json")
    return errors
