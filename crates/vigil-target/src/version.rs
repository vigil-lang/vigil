//! this module defines Linux kernel versions, as a manifest writes them

use std::fmt;
use std::str::FromStr;

/// a kernel release, with major and minor number only
///
/// We drop the patch number because it never changes what we rely on in
/// the verifier. The derived order is numeric, major first: `5.15` comes
/// after `5.8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelVersion {
    pub major: u32,
    pub minor: u32,
}

impl KernelVersion {
    pub const fn new(major: u32, minor: u32) -> Self {
        KernelVersion { major, minor }
    }
}

/// the manifest form: `linux-6.1`
impl fmt::Display for KernelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "linux-{}.{}", self.major, self.minor)
    }
}

/// the error for a string that is not a kernel version
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseVersionError;

impl fmt::Display for ParseVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected a kernel version such as `linux-6.1` or `6.1`")
    }
}

impl std::error::Error for ParseVersionError {}

impl FromStr for KernelVersion {
    type Err = ParseVersionError;

    /// accepts `linux-X.Y`, `X.Y` and `X.Y.Z`, ignoring the patch number.
    /// Each part must be ASCII digits only (no sign, no `v`) and fit in a
    /// `u32`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let digits = s.strip_prefix("linux-").unwrap_or(s);
        let parts: Vec<&str> = digits.split('.').collect();
        // we check the digits by hand because `u32::from_str` accepts a
        // leading `+`
        let numeric = |p: &&str| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit());
        if !(2..=3).contains(&parts.len()) || !parts.iter().all(numeric) {
            return Err(ParseVersionError);
        }
        let number = |p: &str| p.parse().map_err(|_| ParseVersionError);
        Ok(KernelVersion {
            major: number(parts[0])?,
            minor: number(parts[1])?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_accepted_forms() {
        assert_eq!("linux-6.1".parse(), Ok(KernelVersion::new(6, 1)));
        assert_eq!("5.15".parse(), Ok(KernelVersion::new(5, 15)));
        assert_eq!("6.8.12".parse(), Ok(KernelVersion::new(6, 8)));
    }

    #[test]
    fn rejects_malformed_versions() {
        for bad in [
            "", "linux", "linux-6", "6.", ".1", "6.1.", "6.1.2.3", "v6.1", "6.x", "+6.1",
        ] {
            assert_eq!(
                bad.parse::<KernelVersion>(),
                Err(ParseVersionError),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn orders_numerically() {
        assert!(KernelVersion::new(5, 15) > KernelVersion::new(5, 8));
        assert!(KernelVersion::new(6, 0) > KernelVersion::new(5, 19));
    }

    #[test]
    fn displays_with_prefix() {
        assert_eq!(KernelVersion::new(6, 1).to_string(), "linux-6.1");
    }
}
