//! this module defines target profiles: the targeted kernels, the features
//! code may use and the verifier limits. A program must load on every
//! targeted kernel, so features and limits come from the oldest one (see
//! [`Profile::baseline`]). Adding a newer kernel never enables a feature.

use std::fmt;

use crate::arch::Arch;
use crate::version::KernelVersion;

/// the oldest kernel we generate code for. We need bounded loops (5.3) to
/// compile `for` loops without unrolling
pub const MIN_SUPPORTED: KernelVersion = KernelVersion::new(5, 3);

/// a kernel feature that changes what we may generate
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feature {
    /// back edges, accepted when the loop is bounded
    BoundedLoops,
    /// `BPF_MAP_TYPE_RINGBUF` and its helpers
    RingBuf,
    /// `bpf_probe_read_kernel` and its user and string variants
    ProbeReadKernel,
    /// calls of kernel functions (kfuncs)
    Kfuncs,
    /// the `bpf_loop` helper, whose callback is verified once, not once per
    /// iteration
    BpfLoop,
    /// CPU version 4 instructions (`-mcpu=v4`): sign extension, sign
    /// extending loads, signed division
    CpuV4,
}

impl Feature {
    /// all features, in the order the kernel got them
    pub const ALL: [Feature; 6] = [
        Feature::BoundedLoops,
        Feature::ProbeReadKernel,
        Feature::RingBuf,
        Feature::Kfuncs,
        Feature::BpfLoop,
        Feature::CpuV4,
    ];

    /// the first kernel with the feature
    pub const fn since(self) -> KernelVersion {
        match self {
            Feature::BoundedLoops => KernelVersion::new(5, 3),
            Feature::ProbeReadKernel => KernelVersion::new(5, 5),
            Feature::RingBuf => KernelVersion::new(5, 8),
            Feature::Kfuncs => KernelVersion::new(5, 13),
            Feature::BpfLoop => KernelVersion::new(5, 17),
            Feature::CpuV4 => KernelVersion::new(6, 6),
        }
    }

    /// the feature a helper belongs to, when the helper is newer than
    /// [`MIN_SUPPORTED`]. Older helpers are in every kernel we support
    pub const fn of_helper(number: u32) -> Option<Feature> {
        match number {
            // `bpf_probe_read_user`, `bpf_probe_read_kernel` and their
            // string variants
            112..=115 => Some(Feature::ProbeReadKernel),
            // `bpf_ringbuf_output` to `bpf_ringbuf_query`
            130..=134 => Some(Feature::RingBuf),
            181 => Some(Feature::BpfLoop),
            _ => None,
        }
    }

    /// the name used in diagnostics
    pub const fn name(self) -> &'static str {
        match self {
            Feature::BoundedLoops => "bounded-loops",
            Feature::ProbeReadKernel => "probe_read_kernel",
            Feature::RingBuf => "ringbuf",
            Feature::Kfuncs => "kfuncs",
            Feature::BpfLoop => "bpf_loop",
            Feature::CpuV4 => "cpu-v4",
        }
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// the verifier limits on the programs of a profile
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// the most instructions in a program, its length (`lddw` counts as
    /// two): `BPF_COMPLEXITY_LIMIT_INSNS` when privileged, `BPF_MAXINSNS`
    /// otherwise. The verifier also stops after processing 1 million
    /// instructions, which only loading can tell.
    pub max_insns: u32,
    /// the stack of a chain of calls, all its frames together, in bytes
    pub stack_bytes: u32,
    /// the most frames in a chain of BPF-to-BPF calls and callbacks, the
    /// program's own included (`MAX_CALL_FRAMES`)
    pub max_call_depth: u32,
    /// the longest chain of tail calls (`MAX_TAIL_CALL_CNT`)
    pub max_tail_calls: u32,
}

/// the targeted kernels, the privilege and the architecture
///
/// Invariant, kept by [`Profile::new`]: the kernel list is sorted, without
/// duplicates, not empty, and nothing in it is older than [`MIN_SUPPORTED`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    kernels: Vec<KernelVersion>,
    privileged: bool,
    arch: Arch,
}

impl Profile {
    /// creating a profile for the host architecture, sorting and
    /// deduplicating the kernels. `None` if the list is empty or has a
    /// kernel older than [`MIN_SUPPORTED`]
    pub fn new(mut kernels: Vec<KernelVersion>, privileged: bool) -> Option<Self> {
        kernels.sort();
        kernels.dedup();
        match kernels.first() {
            Some(&oldest) if oldest >= MIN_SUPPORTED => Some(Profile {
                kernels,
                privileged,
                arch: Arch::host(),
            }),
            _ => None,
        }
    }

    /// the profile of a lone `.vg` file: Linux 6.1, privileged, host
    /// architecture
    pub fn default_profile() -> Self {
        Profile {
            kernels: vec![KernelVersion::new(6, 1)],
            privileged: true,
            arch: Arch::host(),
        }
    }

    /// the same profile for another architecture
    pub fn with_arch(self, arch: Arch) -> Self {
        Profile { arch, ..self }
    }

    /// the target architecture, the host one by default
    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// the targeted kernels, oldest first
    pub fn kernels(&self) -> &[KernelVersion] {
        &self.kernels
    }

    pub fn privileged(&self) -> bool {
        self.privileged
    }

    /// the oldest targeted kernel, which decides features and limits. Here
    /// is safe to index because the list is sorted and never empty.
    pub fn baseline(&self) -> KernelVersion {
        self.kernels[0]
    }

    /// checking if every targeted kernel has the feature
    pub fn supports(&self, feature: Feature) -> bool {
        self.baseline() >= feature.since()
    }

    /// the features of every targeted kernel, in the order of
    /// [`Feature::ALL`]
    pub fn features(&self) -> impl Iterator<Item = Feature> + '_ {
        Feature::ALL.into_iter().filter(|&f| self.supports(f))
    }

    /// the verifier limits of the baseline kernel
    pub fn limits(&self) -> Limits {
        Limits {
            // since 5.2 a privileged program may have one million
            // instructions. Unprivileged ones keep the old 4096. We never see
            // older kernels because `MIN_SUPPORTED` is 5.3.
            max_insns: if self.privileged { 1_000_000 } else { 4096 },
            stack_bytes: 512,
            max_call_depth: 8,
            // `MAX_TAIL_CALL_CNT` went from 32 to 33 in 5.17
            max_tail_calls: if self.baseline() >= KernelVersion::new(5, 17) {
                33
            } else {
                32
            },
        }
    }
}

/// the kernels separated by commas, then `(unprivileged)` if so
impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<_> = self.kernels.iter().map(ToString::to_string).collect();
        write!(f, "{}", names.join(", "))?;
        if !self.privileged {
            write!(f, " (unprivileged)")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u32, minor: u32) -> KernelVersion {
        KernelVersion::new(major, minor)
    }

    #[test]
    fn baseline_is_the_oldest_kernel() {
        let profile = Profile::new(vec![v(6, 8), v(5, 15), v(6, 1), v(5, 15)], true).unwrap();
        assert_eq!(profile.kernels(), [v(5, 15), v(6, 1), v(6, 8)]);
        assert_eq!(profile.baseline(), v(5, 15));
    }

    #[test]
    fn features_follow_the_baseline() {
        let profile = Profile::new(vec![v(5, 15), v(6, 8)], true).unwrap();
        let features: Vec<_> = profile.features().collect();
        assert_eq!(
            features,
            [
                Feature::BoundedLoops,
                Feature::ProbeReadKernel,
                Feature::RingBuf,
                Feature::Kfuncs
            ]
        );
        assert!(!profile.supports(Feature::BpfLoop));
        assert!(
            Profile::new(vec![v(5, 17)], true)
                .unwrap()
                .supports(Feature::BpfLoop)
        );
    }

    #[test]
    fn helpers_newer_than_the_minimum() {
        assert_eq!(Feature::of_helper(114), Some(Feature::ProbeReadKernel));
        assert_eq!(Feature::of_helper(130), Some(Feature::RingBuf));
        assert_eq!(Feature::of_helper(134), Some(Feature::RingBuf));
        assert_eq!(Feature::of_helper(181), Some(Feature::BpfLoop));
        assert_eq!(Feature::of_helper(14), None);
        assert_eq!(Feature::of_helper(112), Some(Feature::ProbeReadKernel));
        assert_eq!(Feature::of_helper(111), None);
        assert_eq!(Feature::of_helper(135), None);
    }

    #[test]
    fn rejects_empty_and_too_old_profiles() {
        assert_eq!(Profile::new(vec![], true), None);
        assert_eq!(Profile::new(vec![v(4, 19), v(6, 1)], true), None);
        assert!(Profile::new(vec![MIN_SUPPORTED], true).is_some());
    }

    #[test]
    fn limits_depend_on_privileges_and_version() {
        let old = Profile::new(vec![v(5, 10)], false).unwrap().limits();
        assert_eq!(old.max_insns, 4096);
        assert_eq!(old.max_tail_calls, 32);
        let new = Profile::new(vec![v(6, 1)], true).unwrap().limits();
        assert_eq!(new.max_insns, 1_000_000);
        assert_eq!(new.max_tail_calls, 33);
        assert_eq!(new.stack_bytes, 512);
    }
}
