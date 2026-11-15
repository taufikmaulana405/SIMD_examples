//! Deterministic RNG replicating `quad-rand 0.2.3` bit-for-bit.
//!
//! Stage 1 proved the legacy binaries are *accidentally deterministic*:
//! `quad-rand`'s global state starts at zero and nothing ever seeds it unless
//! the harness passes `--seed`. The generator is PCG-XSH-RR with the
//! constants below, verified against the vendored source in
//! `~/.cargo/registry/src/*/quad-rand-0.2.3/src/lib.rs`.
//!
//! Replicating it exactly (instead of a fresh xorshift) means
//! `spawn::create_particles(12345)` produces **bit-identical** initial
//! conditions to `legacy_work` with `--seed 12345`, which is what makes the
//! Stage-3 parity tests meaningful. This is safe, dependency-free Rust.

const MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const INCREMENT: u64 = 1_442_695_040_888_963_407;

/// PCG-XSH-RR generator with `quad-rand 0.2.3` semantics.
///
/// A fresh generator has state zero, exactly like quad-rand's `GLOBAL_STATE`.
#[derive(Debug, Clone)]
pub struct QuadRand {
    state: u64,
}

impl QuadRand {
    /// New generator with the zero initial state quad-rand uses.
    pub fn new() -> Self {
        Self { state: 0 }
    }

    /// Seed like `quad-rand::srand`: reset to zero, advance once, add the
    /// seed (wrapping), advance once more. Consumes two outputs, exactly as
    /// the original.
    pub fn srand(&mut self, seed: u64) {
        self.state = 0;
        self.rand_u32();
        self.state = self.state.wrapping_add(seed);
        self.rand_u32();
    }

    /// Next `u32`, PCG-XSH-RR output function.
    pub fn rand_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// `gen_range(low, high)` for `f32`, replicating quad-rand's
    /// `impl_random_range!(f32, ...)`: one `rand_u32` call, scaled in `f64`,
    /// then narrowed. Returns values in `[low, high)`.
    ///
    /// ```
    /// use gsim_core::rng::QuadRand;
    /// let mut rng = QuadRand::new();
    /// rng.srand(12345);
    /// let a = rng.gen_range_f32(0.0, 1.0);
    /// let mut rng2 = QuadRand::new();
    /// rng2.srand(12345);
    /// assert_eq!(a, rng2.gen_range_f32(0.0, 1.0));
    /// assert!(a >= 0.0 && a < 1.0);
    /// ```
    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        let r = self.rand_u32() as f64 / (u32::MAX as f64 + 1.0);
        let scaled = low as f64 + (high as f64 - low as f64) * r;
        scaled as f32
    }
}

impl Default for QuadRand {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First outputs after `srand(12345)`; guards against accidental drift
    /// from the quad-rand algorithm (values recorded from the vendored
    /// quad-rand 0.2.3 source semantics).
    #[test]
    fn seed_12345_first_outputs() {
        let mut rng = QuadRand::new();
        rng.srand(12345);
        // PCG-XSH-RR with the srand pre-advance: these are the exact first
        // two u32 outputs of quad-rand 0.2.3 after srand(12345).
        let a = rng.rand_u32();
        let b = rng.rand_u32();
        assert_eq!((a, b), (0x5421_840f, 0xbca9_019b));
    }

    #[test]
    fn unseeded_matches_quad_rand_default_state() {
        // quad-rand with no srand starts from state 0; first rand() output
        // is fully determined by the increment constant.
        let mut rng = QuadRand::new();
        let first = rng.rand_u32();
        assert_eq!(first, 0);
    }
}
