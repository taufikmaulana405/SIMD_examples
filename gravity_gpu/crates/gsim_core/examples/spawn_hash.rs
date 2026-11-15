fn main() {
    let p = gsim_core::spawn::create_particles(12345, 10000);
    println!("hash {}", p.positions_hash());
    println!("total {:.6} max {:.6}", p.total_mass(), p.largest_mass());
}
