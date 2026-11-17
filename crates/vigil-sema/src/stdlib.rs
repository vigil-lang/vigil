//! this module embeds the standard library in the compiler. The `std.<name>`
//! modules live in `std/`, next to `compiler/`, and are included with
//! `include_str!`, so we need no search path to find them

/// an entry of [`MODULES`]: the module path, the name shown in
/// diagnostics (`<std>/` and the file) and the text of the file
macro_rules! module {
    ($path:literal, $file:literal) => {
        (
            $path,
            concat!("<std>/", $file),
            include_str!(concat!("../../../../std/", $file)),
        )
    };
}

/// the modules of the standard library: path (`std.bpf`), name for
/// diagnostics and source
pub const MODULES: &[(&str, &str, &str)] = &[
    module!("std.bpf", "bpf.vg"),
    module!("std.debug", "debug.vg"),
    module!("std.fixed", "fixed.vg"),
    module!("std.ml", "ml.vg"),
    module!("std.ml.features", "ml/features.vg"),
    module!("std.ml.tensor", "ml/tensor.vg"),
    module!("std.net", "net.vg"),
    module!("std.sketch", "sketch.vg"),
    module!("std.testing", "testing.vg"),
];
