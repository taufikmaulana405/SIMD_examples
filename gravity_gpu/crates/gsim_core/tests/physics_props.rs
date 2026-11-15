//! Property tests for the CPU physics core (Stage 3, task 1).
//!
//! Conservation laws under merging (mass, momentum, center of mass), force
//! antisymmetry (Newton's third law: total force on the system is zero),
//! translation invariance of the acceleration field, agreement of the
//! `f32` kernel with the `f64` oracle, spawn determinism, and a regression
//! test pinning the legacy merge's single-forward-pass (quirk b) behavior.
//! Pure CPU — no GPU needed.

use gsim_core::cpu::{calculate_accelerations, calculate_accelerations_f64, merge_colliding};
use gsim_core::spawn::{create_particles, ParticleData, Particles};
use proptest::prelude::*;

type Body = (f32, f32, f32, f32, f32, f32); // x, y, vx, vy, mass, brightness

fn assemble(v: Vec<Body>) -> Particles {
    let mut p = Particles::with_capacity(v.len());
    for (x, y, vx, vy, mass, brightness) in v {
        p.push(ParticleData {
            x,
            y,
            vx,
            vy,
            mass,
            brightness,
        });
    }
    p
}

/// Random clouds; `spread` controls the position box half-width. Small
/// spreads guarantee overlap-heavy (merge-heavy) inputs.
fn arb_cloud(spread: f32) -> impl Strategy<Value = Particles> {
    (0usize..=24)
        .prop_flat_map(move |n| {
            prop::collection::vec(
                (
                    -spread..spread,
                    -spread..spread,
                    -25.0f32..25.0f32,
                    -25.0f32..25.0f32,
                    1.0f32..60.0f32,
                    0.55f32..1.0f32,
                ),
                n,
            )
        })
        .prop_map(assemble)
}

/// Total linear momentum in `f64` (avoids polluting the test with the
/// implementation's own `f32` accumulation order).
fn momentum(p: &Particles) -> (f64, f64) {
    let mut px = 0.0f64;
    let mut py = 0.0f64;
    for i in 0..p.len() {
        px += f64::from(p.vx[i]) * f64::from(p.mass[i]);
        py += f64::from(p.vy[i]) * f64::from(p.mass[i]);
    }
    (px, py)
}

fn com(p: &Particles) -> (f64, f64) {
    let (mut wx, mut wy, mut m) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..p.len() {
        let mi = f64::from(p.mass[i]);
        wx += f64::from(p.x[i]) * mi;
        wy += f64::from(p.y[i]) * mi;
        m += mi;
    }
    if m == 0.0 {
        (0.0, 0.0)
    } else {
        (wx / m, wy / m)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Perfectly-inelastic merging must conserve total mass (f32 rounding
    /// only: each merge is one f32 addition of the two masses).
    #[test]
    fn mass_conserved_by_merge(p in arb_cloud(6.0)) {
        let before = p.total_mass();
        let mut q = p.clone();
        merge_colliding(&mut q);
        prop_assert!(q.len() <= p.len());
        let tol = 1e-4 * before.max(1.0);
        prop_assert!(
            (before - q.total_mass()).abs() <= tol,
            "mass {before} -> {}",
            q.total_mass()
        );
    }

    /// Merging must conserve total linear momentum: the merge rule is the
    /// mass-weighted velocity mean, i.e. momentum conservation by
    /// construction. Tolerance is f32-rounding scaled by system size.
    #[test]
    fn momentum_conserved_by_merge(p in arb_cloud(6.0)) {
        let (bx, by) = momentum(&p);
        let mut q = p.clone();
        merge_colliding(&mut q);
        let (ax, ay) = momentum(&q);
        let scale = 1.0 + bx.abs().max(by.abs());
        prop_assert!(
            (bx - ax).abs() <= 1e-4 * scale && (by - ay).abs() <= 1e-4 * scale,
            "momentum ({bx},{by}) -> ({ax},{ay})"
        );
    }

    /// Merging must not move the center of mass (mass-weighted position
    /// mean, same argument as momentum).
    #[test]
    fn com_conserved_by_merge(p in arb_cloud(6.0)) {
        let (bx, by) = com(&p);
        let mut q = p.clone();
        merge_colliding(&mut q);
        if q.is_empty() {
            return Ok(());
        }
        let (ax, ay) = com(&q);
        prop_assert!(
            (bx - ax).abs() <= 1e-3 && (by - ay).abs() <= 1e-3,
            "COM ({bx},{by}) -> ({ax},{ay})"
        );
    }

    /// Newton's third law: the mass-weighted acceleration sum (total force)
    /// must vanish up to f32 summation rounding.
    #[test]
    fn force_antisymmetry(p in arb_cloud(60.0)) {
        let n = p.len();
        let (mut ax, mut ay) = (vec![0.0f32; n], vec![0.0f32; n]);
        calculate_accelerations(&p, &mut ax, &mut ay);
        let (mut fx, mut fy, mut scale) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let m = f64::from(p.mass[i]);
            fx += m * f64::from(ax[i]);
            fy += m * f64::from(ay[i]);
            scale += m * (f64::from(ax[i]).abs() + f64::from(ay[i]).abs());
        }
        let tol = 1e-4 * scale + 1e-6;
        prop_assert!(
            fx.abs() <= tol && fy.abs() <= tol,
            "total force ({fx},{fy}) exceeds tol {tol}"
        );
    }

    /// Accelerations depend only on relative positions: shifting every body
    /// by a constant vector must leave the field (almost) unchanged. The
    /// tolerance accounts for f32 cancellation in the shifted differences.
    #[test]
    fn translation_invariance(p in arb_cloud(60.0)) {
        let n = p.len();
        let (mut ax, mut ay) = (vec![0.0f32; n], vec![0.0f32; n]);
        calculate_accelerations(&p, &mut ax, &mut ay);
        let mut q = p.clone();
        for i in 0..n {
            q.x[i] += 37.5;
            q.y[i] -= 42.25;
        }
        let (mut bx, mut by) = (vec![0.0f32; n], vec![0.0f32; n]);
        calculate_accelerations(&q, &mut bx, &mut by);
        let amax: f32 = ax.iter().chain(ay.iter()).map(|a| a.abs()).fold(0.0, f32::max);
        let tol = 1e-4 * amax + 1e-7;
        for i in 0..n {
            prop_assert!(
                (ax[i] - bx[i]).abs() <= tol && (ay[i] - by[i]).abs() <= tol,
                "body {i}: accel changed by translation beyond {tol}"
            );
        }
    }

    /// The `f32` kernel must track the `f64` oracle closely on ordinary
    /// clouds (sanity bound for P1: `f32` roundoff over ~24 summands).
    #[test]
    fn f32_kernel_tracks_f64_oracle(p in arb_cloud(60.0)) {
        let n = p.len();
        let (mut ax, mut ay) = (vec![0.0f32; n], vec![0.0f32; n]);
        calculate_accelerations(&p, &mut ax, &mut ay);
        let (bx, by) = calculate_accelerations_f64(&p);
        let omax: f64 = bx.iter().chain(by.iter()).map(|a| a.abs()).fold(0.0, f64::max);
        let tol = 1e-3 * omax + 1e-9;
        for i in 0..n {
            let ex = (f64::from(ax[i]) - bx[i]).abs();
            let ey = (f64::from(ay[i]) - by[i]).abs();
            prop_assert!(
                ex <= tol && ey <= tol,
                "body {i}: f32 vs f64 error ({ex},{ey}) exceeds {tol}"
            );
        }
    }

    /// Spawn is deterministic: the same seed gives bitwise-identical clouds.
    #[test]
    fn spawn_determinism(seed in 0u64..=u64::MAX, n in 0usize..=64) {
        let a = create_particles(seed, n);
        let b = create_particles(seed, n);
        prop_assert_eq!(a.len(), n);
        prop_assert_eq!(a.x, b.x);
        prop_assert_eq!(a.y, b.y);
        prop_assert_eq!(a.vx, b.vx);
        prop_assert_eq!(a.vy, b.vy);
        prop_assert_eq!(a.mass, b.mass);
        prop_assert_eq!(a.brightness, b.brightness);
    }
}

/// The legacy merge is a SINGLE forward pass over ascending indices
/// (quirk b), not a fixed-point iteration: after body `first` is
/// finalized, later merges can grow another body into overlap with it
/// and the outer loop never rechecks. This pins the exact behavior on
/// a reproduced case (7 bodies -> 3 survivors; two survivors overlap:
/// dist 8.10 < r0+r2 9.57). If this ever changes, the legacy-order
/// replication (and the GPU serial_parity kernel) changed.
#[test]
fn legacy_merge_is_single_forward_pass() {
    let bodies: Vec<Body> = vec![
        (5.251535, -4.579886, 0.0, 0.0, 31.18859, 0.7),
        (3.131298, -1.524635, 0.0, 0.0, 39.60892, 0.7),
        (5.159189, 4.277589, 0.0, 0.0, 49.78706, 0.7),
        (1.691649, -4.749462, 0.0, 0.0, 11.67141, 0.7),
        (2.240715, 4.591907, 0.0, 0.0, 51.9982, 0.7),
        (-3.322486, 5.068306, 0.0, 0.0, 46.36084, 0.7),
        (-5.158888, -3.038789, 0.0, 0.0, 38.93923, 0.7),
    ];
    let mut q = assemble(bodies);
    merge_colliding(&mut q);
    assert_eq!(q.len(), 3);
    let mut masses: Vec<f32> = (0..3).map(|i| q.mass[i]).collect();
    masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (got, want) in masses.iter().zip([38.93923f32, 82.46892, 148.1461]) {
        assert!((got - want).abs() < 1e-4, "mass {got} != {want}");
    }
    // Two survivors overlap: the single-pass merge left it behind.
    let mut overlap_found = false;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let dx = q.x[j] - q.x[i];
            let dy = q.y[j] - q.y[i];
            let rc = 0.45 * q.mass[i].sqrt() + 0.45 * q.mass[j].sqrt();
            if dx * dx + dy * dy <= rc * rc {
                overlap_found = true;
            }
        }
    }
    assert!(overlap_found, "expected the documented leftover overlap");
}
