//! The statistics the research needs, kept small and testable.

/// Standard normal cumulative distribution function.
///
/// Uses the Chebyshev-fitted complementary error function from Numerical
/// Recipes (`erfcc`, fractional error below 1.2e-7), which is ample for
/// probabilities quoted to the cent.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    // Numerical Recipes erfc, fractional error < 1.2e-7.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        r
    } else {
        2.0 - r
    }
}

/// Clamp a probability away from 0 and 1 so log loss stays finite.
pub fn clamp_p(p: f64) -> f64 {
    p.clamp(1e-4, 1.0 - 1e-4)
}

/// Mean Brier score of probabilities against 0/1 outcomes.
pub fn brier(pairs: &[(f64, bool)]) -> f64 {
    mean(pairs.iter().map(|(p, y)| {
        let y = if *y { 1.0 } else { 0.0 };
        (p - y).powi(2)
    }))
}

/// Mean log loss (natural log) of probabilities against 0/1 outcomes.
pub fn log_loss(pairs: &[(f64, bool)]) -> f64 {
    mean(pairs.iter().map(|(p, y)| {
        let p = clamp_p(*p);
        if *y {
            -p.ln()
        } else {
            -(1.0 - p).ln()
        }
    }))
}

/// Mean of an iterator; 0 when empty.
pub fn mean(xs: impl IntoIterator<Item = f64>) -> f64 {
    let (sum, n) = xs
        .into_iter()
        .fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Sample standard deviation; 0 with fewer than two values.
pub fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs.iter().copied());
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

/// One bucket of a calibration curve.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibrationBin {
    /// Lower edge of the predicted-probability bucket.
    pub lo: f64,
    /// Upper edge.
    pub hi: f64,
    /// Number of predictions in it.
    pub n: usize,
    /// Mean predicted probability.
    pub predicted: f64,
    /// Fraction that resolved yes.
    pub observed: f64,
}

/// Calibration curve over `bins` equal-width buckets; empty buckets omitted.
pub fn calibration(pairs: &[(f64, bool)], bins: usize) -> Vec<CalibrationBin> {
    let bins = bins.max(1);
    (0..bins)
        .filter_map(|i| {
            let lo = i as f64 / bins as f64;
            let hi = (i + 1) as f64 / bins as f64;
            let inside: Vec<_> = pairs
                .iter()
                .filter(|(p, _)| *p >= lo && (*p < hi || (i == bins - 1 && *p <= hi)))
                .collect();
            (!inside.is_empty()).then(|| CalibrationBin {
                lo,
                hi,
                n: inside.len(),
                predicted: mean(inside.iter().map(|(p, _)| *p)),
                observed: mean(inside.iter().map(|(_, y)| if *y { 1.0 } else { 0.0 })),
            })
        })
        .collect()
}

/// Ordinary least squares `y = a + b x`. Returns `(a, b)`; `None` without
/// at least two distinct x values.
pub fn ols(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    if points.len() < 2 {
        return None;
    }
    let mx = mean(points.iter().map(|p| p.0));
    let my = mean(points.iter().map(|p| p.1));
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if sxx <= f64::EPSILON {
        return None;
    }
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let b = sxy / sxx;
    Some((my - b * mx, b))
}

/// Kalshi's taker fee per contract at price `p` (dollars): 7% of p(1-p),
/// rounded up to the cent per contract.
pub fn kalshi_taker_fee(p: f64) -> f64 {
    (0.07 * p * (1.0 - p) * 100.0).ceil() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_cdf_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((norm_cdf(1.0) - 0.841_344_75).abs() < 1e-6);
        assert!((norm_cdf(-1.96) - 0.024_997_9).abs() < 1e-6);
        assert!((norm_cdf(3.0) - 0.998_650_1).abs() < 1e-6);
    }

    #[test]
    fn test_scores_reward_confident_correct_forecasts() {
        let good = [(0.9, true), (0.1, false)];
        let bad = [(0.1, true), (0.9, false)];
        assert!(brier(&good) < brier(&bad));
        assert!(log_loss(&good) < log_loss(&bad));
        assert!((brier(&[(0.5, true), (0.5, false)]) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_log_loss_stays_finite_at_certainty() {
        assert!(log_loss(&[(1.0, false)]).is_finite());
        assert!(log_loss(&[(0.0, true)]).is_finite());
    }

    #[test]
    fn test_calibration_buckets_predictions() {
        let pairs = [(0.05, false), (0.15, false), (0.95, true), (1.0, true)];
        let curve = calibration(&pairs, 10);
        assert_eq!(curve.len(), 3);
        assert_eq!(curve.last().unwrap().n, 2);
        assert_eq!(curve.last().unwrap().observed, 1.0);
    }

    #[test]
    fn test_ols_recovers_a_line() {
        let pts: Vec<_> = (0..10).map(|i| (i as f64, 2.0 + 3.0 * i as f64)).collect();
        let (a, b) = ols(&pts).unwrap();
        assert!((a - 2.0).abs() < 1e-9 && (b - 3.0).abs() < 1e-9);
        assert!(ols(&[(1.0, 1.0), (1.0, 2.0)]).is_none());
    }

    #[test]
    fn test_kalshi_fee_matches_schedule() {
        // 0.07 * 0.5 * 0.5 = 0.0175 -> 2 cents.
        assert!((kalshi_taker_fee(0.5) - 0.02).abs() < 1e-12);
        assert!((kalshi_taker_fee(0.99) - 0.01).abs() < 1e-12);
    }
}
