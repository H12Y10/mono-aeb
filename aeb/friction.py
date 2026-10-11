"""Road-friction aware effective braking deceleration (measured table).

Background: the AEB trigger threshold TTC_AEB(v) = t_react + v / (2 * a_max)
uses a_max, a dry-road design value. On low-friction surfaces (wet / ice-like)
the achievable deceleration is much lower (measured: about 7.1 m/s^2 at
mu = 3.5 versus about 2.5 m/s^2 at mu = 0.4), so a fixed a_max makes the
threshold trigger too late. This module replaces a_max with a measured
a_eff(mu), so the trigger threshold adapts to the actual road surface.

Measurement protocol (2026-10, CARLA simulator):
  - a_eff(mu) = v0b^2 / (2 * d_stop): v0b is the measured speed at brake onset
    and d_stop the trapezoidal integral of the speed trace over the braking
    phase;
  - design point v0 = 20 m/s: the vehicle accelerates from standstill at full
    throttle up to v0 - 0.2, then applies full braking (throttle = 0,
    brake = 1.0) until standstill;
  - straight road, synchronous mode with fixed_delta = 1/30 s, clear rendering
    (rendering weather does not change tyre friction); repeat runs at mu = 0.4
    and 1.0 reproduced identical traces, so one run per mu was sufficient;
  - the mu = 0.6 point was re-measured at an obstacle-free start position after
    the first run clipped roadside infrastructure;
  - coverage is mu in [0.4, 3.5]: values outside the range clamp to the nearest
    endpoint (no extrapolation), and the result is capped at a_max (from
    mu ~ 2.7 upward a_eff reaches a_max, so the behaviour becomes identical to
    the fixed-a_max formula).
"""

from itertools import pairwise

# mu -> measured effective braking deceleration a_eff (m/s^2), ascending in mu.
# Values are the measured per-run decelerations (frozen 2026-10).
A_EFF_TABLE: tuple[tuple[float, float], ...] = (
    (0.4, 2.511),
    (0.5, 3.057),
    (0.6, 3.566),
    (0.8, 4.751),
    (1.0, 5.195),
    (1.2, 5.639),
    (1.5, 6.376),
    (2.0, 6.501),
    (3.0, 7.018),
    (3.5, 7.118),
)


def a_eff_for_mu(mu: float, a_max: float) -> float:
    """Piecewise-linear interpolation over the measured table, capped at a_max.

    Values outside the table range clamp to the nearest endpoint (no
    extrapolation). NaN raises ValueError explicitly: it must not silently fall
    through to the high-friction end, which would give the least conservative
    threshold.
    """
    if mu != mu:  # NaN
        raise ValueError("friction_mu must not be NaN")
    mus = A_EFF_TABLE
    a_eff = mus[-1][1]
    if mu <= mus[0][0]:
        a_eff = mus[0][1]
    elif mu < mus[-1][0]:
        for (m0, a0), (m1, a1) in pairwise(mus):
            if m0 <= mu <= m1:
                a_eff = a0 + (mu - m0) / (m1 - m0) * (a1 - a0)
                break
    return min(a_eff, a_max)
