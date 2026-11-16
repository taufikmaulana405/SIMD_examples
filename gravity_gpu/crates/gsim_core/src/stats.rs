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
    /// Trail points currently stored (`min(head, ring capacity)`),
    /// including expired ones the renderer culls by age.
    pub trail_points: u32,
    /// Trail points still within the length limit (unexpired) — the
    /// count the legacy HUD showed, since its trail deque physically
    /// removed expired points.
    pub trail_live: u32,
    /// Ring slot of the oldest live trail point. The live points form
    /// the contiguous arc `[trail_start, trail_start + trail_live)`
    /// (wrapping at the ring capacity), which is all the renderer draws.
    pub trail_start: u32,
}

/// Number of `u32` words in the GPU stats buffer.
pub const STATS_WORDS: usize = 8;

impl Stats {
    /// Decode from the eight raw words of the GPU stats buffer.
    ///
    /// ```
    /// use gsim_core::stats::Stats;
    /// let words = [10u32, 0x447a0000, 0x41200000, 0x0, 0x0, 500, 320, 12];
    /// let s = Stats::from_words(&words);
    /// assert_eq!(s.count, 10);
    /// assert_eq!(s.trail_points, 500);
    /// assert_eq!(s.trail_live, 320);
    /// assert_eq!(s.trail_start, 12);
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
            trail_live: if words.len() >= 7 { words[6] } else { 0 },
            trail_start: if words.len() >= 8 { words[7] } else { 0 },
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
            self.trail_live,
            self.trail_start,
        ]
    }
}
