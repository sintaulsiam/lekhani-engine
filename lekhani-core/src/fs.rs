//! Safe atomic filesystem operations with secure permissions

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Safely write bytes to a file using process-unique temporary files and atomic rename.
/// On Unix, files are created with restricted 0600 permissions.
pub fn atomic_write_secure<P: AsRef<Path>>(path: P, data: &[u8]) -> Result<(), std::io::Error> {
    let path = path.as_ref();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;

    let file_stem = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("file");
    let pid = std::process::id();
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = parent.join(format!(".{}.tmp.{}.{}", file_stem, pid, count));

    let write_res = (|| {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        opts.mode(0o600);

        let mut file = opts.open(&tmp_path)?;
        file.write_all(data)?;
        file.flush()?;
        Ok::<(), std::io::Error>(())
    })();

    if let Err(e) = write_res {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_write_secure() {
        let temp_dir = std::env::temp_dir().join(format!("lekhani_fs_test_{}", std::process::id()));
        let file_path = temp_dir.join("test_file.bin");

        atomic_write_secure(&file_path, b"secure payload").expect("Write should succeed");
        assert_eq!(std::fs::read(&file_path).unwrap(), b"secure payload");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = std::fs::metadata(&file_path).unwrap();
            let mode = meta.permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "File permissions must be 0600 on Unix");
        }

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
