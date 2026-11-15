//! The tiny per-frame stats struct.
//!
//! This is the **only** buffer the CPU reads back every frame (32 bytes,
//! asynchronously mapped). It feeds the HUD and the per-frame dispatch
//! sizing. Layout mirrors the GPU `stats` storage buffer: eight `u32`s,
//! floats transported as bit patterns.

/// HUD stats: count, total mass, largest mass, center of mass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stats {
    /// Live bodies.
    pub count: u32,
    /// Total mass of the system.
    pub total_mass: f32,
    /// Largest single body.
    pub max_mass: f32,
    /// Center of mass x.
    pub com_x: f32,
    /// Center of mass y.
    pub com_y: f32,
    /// Trail points currently stored (`min(head, TRAIL_MAX_POINTS)`).
    pub trail_points: u32,
}

/// Number of `u32` words in the GPU stats buffer.
pub const STATS_WORDS: usize = 8;

impl Stats {
    /// Decode from the eight raw words of the GPU stats buffer.
    ///
    /// ```
    /// use gsim_core::stats::Stats;
    /// let words = [10u32, 0x447a0000, 0x41200000, 0x0, 0x0, 500, 0, 0];
    /// let s = Stats::from_words(&words);
    /// assert_eq!(s.count, 10);
    /// assert!((s.total_mass - 1000.0).abs() < 0.01);
    /// ```
    pub fn from_words(words: &[u32]) -> Self {
        assert!(words.len() >= 6);
        Self {
            count: words[0],
            total_mass: f32::from_bits(words[1]),
            max_mass: f32::from_bits(words[2]),
            com_x: f32::from_bits(words[3]),
            com_y: f32::from_bits(words[4]),
            trail_points: words[5],
        }
    }

    /// Encode into eight raw words (for tests / CPU-side construction).
    pub fn to_words(&self) -> [u32; STATS_WORDS] {
        [
            self.count,
            self.total_mass.to_bits(),
            self.max_mass.to_bits(),
            self.com_x.to_bits(),
            self.com_y.to_bits(),
            self.trail_points,
            0,
            0,
        ]
    }
}
