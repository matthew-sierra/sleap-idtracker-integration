"""Invariant tests for the numerical core of the SLEAP->idtracker.ai port.

These are PROPERTY tests, not example tests: each asserts something that must
hold for every input, and each is swept across the SCALES the pipeline actually
runs at. Every bug found in the port so far was an invariant violation that
only appeared above a scale the ad-hoc checks never reached.

Run:  python tests/test_numerics.py        (no pytest needed)
      pytest tests/test_numerics.py        (when pytest is available)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "idtrackerai" / "src"))

# The scales the pipeline is actually run at. Saturation appears only above a
# ~53-vote margin, which no small fixture reaches -- hence the sweep.
N_ANIMALS = [2, 5, 10, 16]
FRAG_LEN = [4, 20, 53, 60, 500, 5000]


def p1_from_frequencies(freq):
    """Mirror of fragment.set_P1_from_frequencies (fragment.py:483-491)."""
    freq = np.asarray(freq, dtype=float)
    with np.errstate(over="ignore"):
        return 1.0 / (2.0 ** (freq[:, None] - freq[None, :])).sum(axis=0)


# ---------------------------------------------------------------- P1 ------
def test_p1_is_a_distribution():
    """P1 must sum to 1 and be non-negative, at every scale."""
    for n in N_ANIMALS:
        for L in FRAG_LEN:
            freq = np.zeros(n); freq[0] = L
            p1 = p1_from_frequencies(freq)
            assert np.all(p1 >= 0), f"n={n} L={L}: negative P1"
            assert abs(p1.sum() - 1.0) < 1e-9, f"n={n} L={L}: sums to {p1.sum()}"


def test_p1_never_reaches_exactly_one():
    """THE SATURATION INVARIANT.

    P1 = 2^f_i / sum(2^f_j) is algebraically < 1 whenever any other identity
    has a vote, so `1 - P1` must never be exactly 0. compute_P2_vector
    multiplies by (1 - P1_coexisting); an exact 0 turns a soft discount into an
    irrevocable veto and can zero a fragment's whole P2 vector.
    """
    bad = []
    for n in N_ANIMALS:
        for L in FRAG_LEN:
            freq = np.zeros(n); freq[0] = L; freq[1] = 1   # runner-up has a vote
            p1 = p1_from_frequencies(freq)
            if (1.0 - p1.max()) == 0.0:
                bad.append((n, L, p1.max()))
    assert not bad, (
        "1 - P1 == 0.0 exactly at these (n_animals, fragment_length): "
        + ", ".join(f"({n},{L})" for n, L, _ in bad)
    )


# ---------------------------------------------------------------- P2 ------
def p2(own, coexisting):
    coex = np.asarray(coexisting, dtype=float)
    num = own * (np.prod(1.0 - coex, axis=0) if len(coex) else np.ones_like(own))
    den = num.sum()
    return num / den if den != 0 else np.zeros_like(own)


def test_p2_is_never_all_zero():
    """A valid P1 and N-1 coexisting claimants must leave one slot free.

    With N animals at most N-1 others coexist, so exactly one identity is
    always unclaimed. An all-zero P2 is a failure state, but assign_identity
    (fragment.py:428-433) reads it as an N-way tie and silently returns
    identity 0 -- a failure indistinguishable from a legitimate answer.
    """
    bad = []
    for n in N_ANIMALS:
        for L in FRAG_LEN:
            freq = np.zeros(n); freq[n - 1] = L; freq[0] = 1
            own = p1_from_frequencies(freq)
            coex = []
            for k in range(n - 1):                      # the other N-1 animals
                f = np.zeros(n); f[k] = L; f[(k + 1) % n] = 1
                coex.append(p1_from_frequencies(f))
            v = p2(own, coex)
            if v.sum() == 0.0:
                bad.append((n, L))
    assert not bad, (
        "P2 collapsed to all-zeros (-> identity 0) at (n_animals, length): "
        + ", ".join(map(str, bad))
    )


def test_p2_preserves_veto_ordering():
    """An identity vetoed once must outrank one vetoed six times.

    This is what P2 is FOR. Exact zeros destroy it: 0**1 == 0**6.
    """
    n = 5
    own = p1_from_frequencies([1, 1, 1, 1, 500])
    coex = []
    for ident, count in {0: 6, 1: 5, 2: 3, 3: 2, 4: 1}.items():
        for _ in range(count):
            f = np.zeros(n); f[ident] = 500; f[(ident + 1) % n] = 1
            coex.append(p1_from_frequencies(f))
    v = p2(own, coex)
    assert v.sum() > 0, "P2 collapsed; ordering cannot be checked"
    assert int(np.argmax(v)) == 4, (
        f"identity 5 is vetoed once and should win; argmax was {np.argmax(v) + 1}"
    )


# ------------------------------------------------- cost-matrix sentinels ---
def test_unmatched_cost_is_distinguishable_from_a_real_pair():
    """UNMATCHED_COST must not collide with the cost of a real pairing.

    Cost is 1 - S with S in [0,1], so a zero-overlap pair costs exactly 1.0.
    If UNMATCHED_COST is also 1.0, "no partner" and "worst possible partner"
    are the same number and the solver can move an orphan row for free -- which
    is what made isolated, cleanly-tracked animals get flagged as overlapping.
    """
    import config
    worst_real = 1.0 - 0.0
    assert config.UNMATCHED_COST != worst_real, (
        f"UNMATCHED_COST ({config.UNMATCHED_COST}) equals the cost of a "
        "zero-similarity real pair; padding is indistinguishable from a bad match"
    )


# --------------------------------------------------------------- runner ---
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {t.__name__}\n          {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
