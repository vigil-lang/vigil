//! printing the program generated from a fuzzing seed, as Vigil source or
//! as Core JSON with `--json`:
//!
//! ```text
//! cargo run -p vigil-core --example print_seed SEED [--json]
//! ```
//!
//! we mix the seed with a constant, so seed 0 is not the state 0 (xorshift
//! gives only zeros there)

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let mut rng = vigil_core::generate::Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491));
    let core = vigil_core::generate::program(&mut rng, Default::default());
    if std::env::args().any(|a| a == "--json") {
        println!("{}", serde_json::to_string(&core).unwrap());
    } else {
        print!("{}", vigil_core::print::print(&core));
    }
}
