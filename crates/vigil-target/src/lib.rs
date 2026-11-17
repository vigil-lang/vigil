//! this crate defines the kernel targets of a project: the kernels it
//! supports, their features and verifier limits, and the `vigil.toml`
//! manifest that declares them.
//!
//! [`parse_manifest`] reads a [`Manifest`]. The later phases ask its
//! [`Profile`] ([`Profile::supports`], [`Profile::limits`]) what they may
//! generate.

mod arch;
pub mod manifest;
mod profile;
mod version;

pub use arch::Arch;
pub use manifest::{Dependency, GitRef, MANIFEST_NAME, Manifest, Source, parse_manifest};
pub use profile::{Feature, Limits, MIN_SUPPORTED, Profile};
pub use version::{KernelVersion, ParseVersionError};
