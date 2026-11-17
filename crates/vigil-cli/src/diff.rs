//! this module computes the line diffs that `vigil fmt --diff` prints. We
//! use a full LCS table, quadratic in time and memory, but it is fine for a
//! source file and its formatted version.

/// the unified diff of `old` and `new`, with three lines of context, empty
/// when they are equal. The new side is named `name (formatted)`. Changes
/// close enough share a hunk, so no context line is printed twice.
pub fn unified(old: &str, new: &str, name: &str) -> String {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    // `lcs[i][j]` is the LCS length of `a[i..]` and `b[j..]`. We fill it
    // from the ends so the walk below can go forwards
    let mut lcs = vec![vec![0u32; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    // the edit script, as `(kind, line of a, line of b)`
    let mut ops = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            ops.push((' ', i, j));
            i += 1;
            j += 1;
        } else if i < a.len() && (j == b.len() || lcs[i + 1][j] >= lcs[i][j + 1]) {
            // on a tie we remove first, so a changed block shows all old
            // lines and then all new ones
            ops.push(('-', i, j));
            i += 1;
        } else {
            ops.push(('+', i, j));
            j += 1;
        }
    }
    const CONTEXT: usize = 3;
    let changed: Vec<usize> = (0..ops.len()).filter(|&k| ops[k].0 != ' ').collect();
    if changed.is_empty() {
        return String::new();
    }
    let mut out = format!("--- {name}\n+++ {name} (formatted)\n");
    let mut k = 0;
    while k < changed.len() {
        let start = changed[k].saturating_sub(CONTEXT);
        let mut end = changed[k];
        while k < changed.len() && changed[k] <= end + 2 * CONTEXT {
            end = changed[k];
            k += 1;
        }
        let end = (end + CONTEXT + 1).min(ops.len());
        let hunk = &ops[start..end];
        let old_count = hunk.iter().filter(|o| o.0 != '+').count();
        let new_count = hunk.iter().filter(|o| o.0 != '-').count();
        out.push_str(&format!(
            "@@ -{},{old_count} +{},{new_count} @@\n",
            hunk[0].1 + 1,
            hunk[0].2 + 1
        ));
        for &(kind, i, j) in hunk {
            let line = if kind == '+' { b[j] } else { a[i] };
            out.push(kind);
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! testing the hunks of a one-line change and of a split line
    use super::unified;

    #[test]
    fn shows_changed_lines_with_context() {
        let old = "a\nb\nc\nd\ne\nf\ng\nh\n";
        let new = "a\nb\nc\nD\ne\nf\ng\nh\n";
        assert_eq!(
            unified(old, new, "x.vg"),
            "--- x.vg\n+++ x.vg (formatted)\n@@ -1,7 +1,7 @@\n a\n b\n c\n-d\n+D\n e\n f\n g\n"
        );
        assert_eq!(unified(old, old, "x.vg"), "");
    }

    #[test]
    fn shows_inserted_lines() {
        let diff = unified("f(a, b)\n", "f(\n    a,\n    b,\n)\n", "y.vg");
        assert!(
            diff.contains("-f(a, b)\n+f(\n+    a,\n+    b,\n+)\n"),
            "{diff}"
        );
    }
}
