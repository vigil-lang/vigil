//! this module runs clang to turn LLVM IR into an eBPF object. The IR goes
//! in on standard input and the object comes out on standard output, so we
//! write no temporary file.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// the oldest clang whose IR syntax we use (opaque pointers)
pub const MIN_VERSION: u32 = 15;

/// a clang that can compile the generated IR
#[derive(Clone, Debug)]
pub struct Clang {
    /// what to run: a path, or a name looked up on the `PATH`
    pub path: PathBuf,
    /// the major version, at least [`MIN_VERSION`]
    pub version: u32,
}

impl Clang {
    /// finding the clang of `$VIGIL_CLANG`, or `clang` on the `PATH`, and
    /// its version. It fails when it cannot be run, does not say it is
    /// clang, or is older than [`MIN_VERSION`].
    pub fn discover() -> Result<Clang, String> {
        let path =
            std::env::var_os("VIGIL_CLANG").map_or_else(|| PathBuf::from("clang"), PathBuf::from);
        let output = Command::new(&path)
            .arg("--version")
            .output()
            .map_err(|e| format!("could not run `{}`: {e}", path.display()))?;
        let text = String::from_utf8_lossy(&output.stdout);
        let version = text
            .split("clang version ")
            .nth(1)
            .and_then(|rest| rest.split('.').next())
            .and_then(|major| major.trim().parse::<u32>().ok())
            .ok_or_else(|| format!("`{}` does not look like clang", path.display()))?;
        if version < MIN_VERSION {
            return Err(format!(
                "clang {version} is too old; Vigil needs clang {MIN_VERSION} or newer"
            ));
        }
        Ok(Clang { path, version })
    }

    /// compiling IR to an eBPF ELF object for a CPU version (`v3` has 32-bit
    /// jumps, from Linux 5.1). The error is what clang printed, or why it
    /// could not run, or that it ran longer than [`timeout`].
    ///
    /// With `$VIGIL_SAVE_IR` the IR is also written to that file, to look at
    /// what clang failed on.
    pub fn compile(&self, ir: &str, cpu: &str) -> Result<Vec<u8>, ClangError> {
        self.compile_within(ir, cpu, timeout())
    }

    /// compiling as [`Clang::compile`] does, killing clang after `limit`
    pub fn compile_within(
        &self,
        ir: &str,
        cpu: &str,
        limit: Option<Duration>,
    ) -> Result<Vec<u8>, ClangError> {
        if let Some(path) = std::env::var_os("VIGIL_SAVE_IR") {
            let _ = std::fs::write(path, ir);
        }
        let mut child = Command::new(&self.path)
            .args(["-target", "bpfel", &format!("-mcpu={cpu}"), "-O2"])
            .args(["-x", "ir", "-c", "-", "-o", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                ClangError::Failed(format!("could not run `{}`: {e}", self.path.display()))
            })?;
        // here we write and read in threads, so we can kill a clang that
        // hangs. After a kill we do not join them, because a child of clang
        // may still hold the pipes.
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let ir = ir.to_owned();
        let writer = std::thread::spawn(move || stdin.write_all(ir.as_bytes()));
        let read = |mut pipe: Box<dyn Read + Send>| {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = pipe.read_to_end(&mut bytes);
                bytes
            })
        };
        let stdout = read(Box::new(child.stdout.take().expect("stdout is piped")));
        let stderr = read(Box::new(child.stderr.take().expect("stderr is piped")));
        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {}
                Err(e) => return Err(ClangError::Failed(format!("clang failed: {e}"))),
            }
            if limit.is_some_and(|limit| started.elapsed() >= limit) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ClangError::TimedOut(limit.unwrap_or_default().as_secs()));
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let written = writer.join().expect("the writer does not panic");
        let (stdout, stderr) = (
            stdout.join().expect("the reader does not panic"),
            stderr.join().expect("the reader does not panic"),
        );
        if !status.success() || stdout.is_empty() {
            let stderr = String::from_utf8_lossy(&stderr);
            // here we drop the cross-compilation warning of wrapped compilers
            // (Nix), and the stack dump after a backend crash, which only
            // matters to LLVM developers
            let message: Vec<_> = stderr
                .lines()
                .filter(|l| !l.contains("cc-wrapper") && !l.contains("nix-wrapped"))
                .take_while(|l| !l.starts_with("PLEASE submit"))
                .collect();
            let mut message = message.join("\n");
            if let Err(e) = written
                && message.trim().is_empty()
            {
                message = format!("could not write to clang: {e}");
            }
            return Err(ClangError::Failed(message));
        }
        Ok(stdout)
    }
}

/// why clang gave no object
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClangError {
    /// what clang printed, or why it could not run
    Failed(String),
    /// clang ran longer than this many seconds and was killed
    TimedOut(u64),
}

/// the default time clang may take, in seconds
pub const DEFAULT_TIMEOUT: u64 = 120;

/// the time clang may take: `$VIGIL_CLANG_TIMEOUT` seconds, or
/// [`DEFAULT_TIMEOUT`]. `0` means no limit.
pub fn timeout() -> Option<Duration> {
    let seconds = std::env::var("VIGIL_CLANG_TIMEOUT")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_TIMEOUT);
    (seconds > 0).then(|| Duration::from_secs(seconds))
}

// the fake clang is a shell script, so these tests are Unix only
#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn kills_a_clang_that_hangs() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("vigil-clang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clang");
        std::fs::write(&path, "#!/bin/sh\nexec sleep 30\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let clang = Clang { path, version: 20 };
        let started = Instant::now();
        let result = clang.compile_within("", "v3", Some(Duration::from_millis(200)));
        assert_eq!(result, Err(ClangError::TimedOut(0)));
        assert!(started.elapsed() < Duration::from_secs(10));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
