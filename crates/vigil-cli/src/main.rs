//! this is the `vigil` command, the simplified distribution: `check`,
//! `build` (LLVM IR, an object through clang, or the certified backend),
//! `core`, `simulate`, `update`, `fmt` and `parse`. Loading, tests,
//! verification and the language server are in the full Vigil.
//!
//! The pipeline is in `vigil_driver`, here we only do the file system and
//! the terminal: the project (`project`), its libraries (`deps`), the BTF,
//! the output and the diagnostics. Every command but `fmt` collects its
//! errors and `report` prints them at the end. `fmt` reports file by file,
//! so one bad file does not stop the others.

mod deps;
mod diff;
mod project;

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{ArgAction, CommandFactory, Parser, Subcommand, ValueEnum};
use vigil_codegen_llvm::ProgramDesc;
use vigil_codegen_llvm::clang::Clang;
use vigil_diagnostics::{Diagnostic, Diagnostics, Renderer, SourceMap, codes};
use vigil_sema::Analysis;

use crate::project::Project;

// the text of `vigil --version`
const VERSION: &str = concat!(
    "Vigil Programming Language Compiler - version ",
    env!("CARGO_PKG_VERSION")
);

// the command line. The `///` comments of the fields and variants here, in
// `Command` and in `Emit` are the text of `vigil --help`. The comments on
// the types are plain `//` because clap would use them as help too
#[derive(Parser)]
// clap prints its version flag as `vigil <version>`, so the flag is ours:
// it prints `VERSION` alone. The subcommand is then optional for the
// parser, and `main` reports it missing
#[command(
    name = "vigil",
    about = "Compiler for the Vigil eBPF language",
    disable_version_flag = true,
    override_usage = "vigil [OPTIONS] <COMMAND>"
)]
struct Cli {
    /// When to color diagnostics.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto, global = true)]
    color: ColorChoice,

    /// Optimization level of the generated code: 0 turns off the
    /// optimizations of the Vigil generator (LLVM still runs with -O2).
    #[arg(short = 'O', long = "opt-level", default_value_t = 1, global = true,
          value_parser = clap::value_parser!(u8).range(0..=1))]
    opt_level: u8,

    /// Print version.
    #[arg(short = 'V', long, action = ArgAction::SetTrue)]
    version: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

// the subcommands
#[derive(Subcommand)]
enum Command {
    /// Check a project or a single `.vg` file without generating code.
    Check {
        /// Project directory or `.vg` file.
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Show the resolved kernel profile.
        #[arg(short, long)]
        verbose: bool,
    },
    /// Compile a project or a single `.vg` file to eBPF.
    Build {
        /// Project directory or `.vg` file.
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Path of the generated file.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// What to generate.
        #[arg(long, value_enum, default_value_t = Emit::Object)]
        emit: Emit,
        /// Compile the XDP and TC programs of Vigil Core with the certified
        /// backend (`vigil-codegen-bpf`), without LLVM nor clang.
        #[arg(long)]
        certified: bool,
    },
    /// Print an XDP or TC program as Vigil Core.
    Core {
        /// Project directory or `.vg` file.
        #[arg(default_value = ".")]
        path: PathBuf,
        /// The program, if there are several.
        #[arg(long)]
        program: Option<String>,
        /// Print JSON instead of the readable form.
        #[arg(long)]
        json: bool,
    },
    /// Run a program of Vigil Core on a packet, in an eBPF interpreter (no
    /// kernel, no root), with the code of the certified backend.
    Simulate {
        /// Project directory or `.vg` file.
        path: PathBuf,
        /// The packet in hexadecimal (`aabbcc...`, spaces allowed).
        #[arg(long)]
        packet: String,
        /// The program, if there are several.
        #[arg(long)]
        program: Option<String>,
    },
    /// Resolve the git dependencies again, from their branches and tags,
    /// fetch them, and write the commits to `vigil.lock`.
    Update {
        /// Project directory or `.vg` file.
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Format `.vg` files in place, keeping their comments.
    Fmt {
        /// Projects, directories or `.vg` files.
        #[arg(default_value = ".")]
        paths: Vec<PathBuf>,
        /// Only report the files that are not formatted (and fail).
        #[arg(long)]
        check: bool,
        /// Show how the files would change, without changing them.
        #[arg(long, conflicts_with = "check")]
        diff: bool,
    },
    /// Parse a `.vg` file and print it in canonical form (debugging aid).
    Parse {
        file: PathBuf,
        /// Print the concrete syntax tree instead.
        #[arg(long)]
        tree: bool,
        /// With `--tree`, include whitespace and comments.
        #[arg(long, requires = "tree")]
        trivia: bool,
    },
}

// what `vigil build` writes
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Emit {
    /// An eBPF ELF object (`.o`).
    Object,
    /// LLVM IR (`.ll`); does not need clang.
    Ir,
    /// The eBPF instructions of each program (`.s`).
    Asm,
}

// when diagnostics are colored
#[derive(Clone, Copy, ValueEnum)]
enum ColorChoice {
    Auto,
    Always,
    Never,
}

impl ColorChoice {
    /// whether to color. `Auto` colors only a terminal, and honors
    /// `NO_COLOR`
    fn enabled(self) -> bool {
        match self {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => {
                io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
            }
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.version {
        println!("{VERSION}");
        return ExitCode::SUCCESS;
    }
    let Some(command) = &cli.command else {
        Cli::command()
            .error(
                clap::error::ErrorKind::MissingSubcommand,
                "a command is required (`vigil --help` lists them)",
            )
            .exit();
    };
    let color = cli.color.enabled();
    let optimize = cli.opt_level > 0;
    let mut sources = SourceMap::new();
    let mut diags = Diagnostics::new();

    match command {
        Command::Check { path, verbose } => {
            if let Some((project, _)) = analyze(path, &mut sources, &mut diags) {
                print_checked(&project, *verbose);
            }
        }
        Command::Build {
            path,
            output,
            emit,
            certified,
        } => {
            // clap cannot forbid one value of `--emit` with `--certified`
            // (`conflicts_with` would forbid every `--emit`), so here we
            // check it before the analysis, with a diagnostic
            if *certified && *emit == Emit::Ir {
                diags.push(
                    Diagnostic::error("`--certified` cannot emit LLVM IR")
                        .with_code(codes::INVALID_OPTIONS)
                        .with_note("the certified backend does not use LLVM, so it has no IR")
                        .with_help("use `--emit asm` or `--emit object` with `--certified`"),
                );
            } else if let Some(compiled) = if *certified {
                compile_certified(path, &mut sources, &mut diags)
            } else {
                compile(path, *emit, optimize, &mut sources, &mut diags)
            } {
                write_output(&compiled, path, output.as_deref(), *emit, &mut diags);
            }
        }
        Command::Core {
            path,
            program,
            json,
        } => {
            if let Some((_, analysis)) = analyze(path, &mut sources, &mut diags) {
                print_core(&analysis, program.as_deref(), *json, &mut diags);
            }
        }
        Command::Simulate {
            path,
            packet,
            program,
        } => {
            // a bad packet is reported before the analysis, which is slower
            if let Some(packet) = parse_packet(packet, &mut diags)
                && let Some((_, analysis)) = analyze(path, &mut sources, &mut diags)
            {
                simulate(&analysis, program.as_deref(), packet, &mut diags);
            }
        }
        Command::Update { path } => {
            if let Some(project) = project::load(path, true, &mut sources, &mut diags) {
                print_libraries(&project);
            }
        }
        Command::Fmt { paths, check, diff } => return format_files(paths, *check, *diff),
        Command::Parse { file, tree, trivia } => {
            if let Some(parse) = parse_file(file, &mut sources, &mut diags) {
                if *tree {
                    print!("{}", parse.debug_tree(*trivia));
                } else {
                    print!("{}", vigil_syntax::pretty(&parse.tree()));
                }
            }
        }
    }

    report(&sources, &diags, color)
}

/// loading, parsing and analyzing a project with its libraries. `None` if
/// there is an error
fn analyze(
    path: &Path,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<(Project, Analysis)> {
    let project = project::load(path, false, sources, diags)?;
    // we read the BTF only if the text mentions `kernel`. A mention in a
    // comment reads it for nothing, but no import is missed
    let imports_kernel = project
        .files
        .iter()
        .any(|(file, _)| sources.get(*file).text().contains("kernel"));
    let btf = if imports_kernel {
        load_btf(&project, diags)
    } else {
        None
    };
    // the standard library goes after the project files, so their
    // diagnostics come first
    let std = match &project.libraries.std {
        Some(std) => std.clone(),
        None => vigil_driver::embedded_std(sources),
    };
    let modules: Vec<_> = project
        .files
        .iter()
        .chain(&project.libraries.modules)
        .cloned()
        .collect();
    let analysis = vigil_driver::analyze(&modules, &std, &project.profile, btf, sources, diags)?;
    if diags.has_errors() {
        return None;
    }
    Some((project, analysis))
}

/// loading the BTF of the manifest, or else of the running kernel. Only
/// the first must exist, the running one may be missing (in a container,
/// on another OS).
fn load_btf(project: &Project, diags: &mut Diagnostics) -> Option<std::sync::Arc<vigil_btf::Btf>> {
    let path = project
        .btf
        .clone()
        .unwrap_or_else(|| PathBuf::from(vigil_btf::VMLINUX));
    // here we read the file ourselves because the note gives only the reason,
    // the message already names the file
    let btf = std::fs::read(&path)
        .map_err(|e| e.to_string())
        .and_then(|bytes| vigil_btf::Btf::parse(&bytes).map_err(|e| e.to_string()));
    match btf {
        Ok(btf) => Some(std::sync::Arc::new(btf)),
        Err(reason) if project.btf.is_some() => {
            diags.push(
                Diagnostic::error(format!("could not read the BTF `{}`", path.display()))
                    .with_code(codes::IO_ERROR)
                    .with_note(reason),
            );
            None
        }
        // without BTF an import from `kernel` fails at the import, which is
        // a better error than a missing file
        Err(_) => None,
    }
}

/// a project compiled by either backend, ready to be written
struct Compiled {
    project: Project,
    programs: Vec<ProgramDesc>,
    /// empty for the certified backend, which does not use LLVM
    ir: String,
    /// empty for `--emit ir`, which does not run clang
    object: Vec<u8>,
    /// the code of each program, in the order of `programs`, empty for
    /// `--emit ir`
    code: Vec<Vec<u8>>,
}

/// building the object of a package with clang, with the errors of the
/// generator or of clang in `diags`
fn build_object(
    analysis: &Analysis,
    sources: &SourceMap,
    options: vigil_codegen_llvm::Options,
    clang: &Clang,
    diags: &mut Diagnostics,
) -> Option<vigil_codegen_llvm::Built> {
    match vigil_codegen_llvm::build(analysis, sources, options, clang) {
        Ok(built) => Some(built),
        Err(vigil_codegen_llvm::BuildError::Diagnostics(errors)) => {
            diags.extend(errors);
            None
        }
        Err(vigil_codegen_llvm::BuildError::Clang(err)) => {
            diags.push(
                Diagnostic::error("clang could not compile the generated code")
                    .with_code(codes::TOOLCHAIN)
                    .with_note(err)
                    .with_note(format!(
                        "unless the program exceeds a limit of eBPF (its stack or its size), \
                         this is a bug in Vigil: please report it at {} with the program and this output",
                        vigil_diagnostics::ISSUES
                    )),
            );
            None
        }
        Err(vigil_codegen_llvm::BuildError::ClangTimeout(seconds)) => {
            diags.push(
                Diagnostic::error(format!("clang did not finish within {seconds} s"))
                    .with_code(codes::TOOLCHAIN)
                    .with_note(
                        "this usually comes from a very large function: a long `@unroll` loop, \
                         or large locals",
                    )
                    .with_help(
                        "make the program smaller, or give clang more time with \
                         `VIGIL_CLANG_TIMEOUT=<seconds>` (0 for no limit)",
                    ),
            );
            None
        }
    }
}

/// compiling a project through LLVM: to IR for `--emit ir` (the IR of
/// the object when clang is found), otherwise to an object by clang,
/// checking the profile limits
fn compile(
    path: &Path,
    emit: Emit,
    optimize: bool,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<Compiled> {
    let (project, analysis) = analyze(path, sources, diags)?;
    let options = vigil_driver::codegen_options(&project.profile, optimize);
    if emit == Emit::Ir {
        // with clang we build the object and give its IR, so a program that
        // needs more stack shows the second pass, with its locals in
        // scratch. Without clang we give the first pass, because the stack
        // is measured on the object
        if let Ok(clang) = Clang::discover() {
            let built = build_object(&analysis, sources, options, &clang, diags)?;
            return Some(Compiled {
                project,
                programs: built.output.programs,
                ir: built.output.ir,
                object: Vec::new(),
                code: Vec::new(),
            });
        }
        return match vigil_codegen_llvm::generate_with(&analysis, sources, options) {
            Ok(output) => Some(Compiled {
                project,
                programs: output.programs,
                ir: output.ir,
                object: Vec::new(),
                code: Vec::new(),
            }),
            Err(errors) => {
                diags.extend(errors);
                None
            }
        };
    }
    let clang = match Clang::discover() {
        Ok(clang) => clang,
        Err(err) => {
            diags.push(
                Diagnostic::error("clang is needed to build eBPF objects")
                    .with_code(codes::TOOLCHAIN)
                    .with_note(err)
                    .with_help(format!(
                        "install clang {} or newer with the BPF target, or point `VIGIL_CLANG` to it; \
                         `--emit ir` and `--certified` do not need it",
                        vigil_codegen_llvm::clang::MIN_VERSION
                    )),
            );
            return None;
        }
    };
    let built = build_object(&analysis, sources, options, &clang, diags)?;
    // checking length, stack and calls against the kernel limits here, so
    // the error points to the program and not to the verifier. We check the
    // length of the program, not the instructions the verifier processes
    // (1 million at most), which are only known when loading.

    let limits = project.profile.limits();
    let mut code = Vec::new();
    for program in &built.output.programs {
        let section = program.info.section();
        let insns = vigil_codegen_llvm::program_code(&built.object, &section, &program.name)
            .unwrap_or_default();
        let stats = vigil_codegen_llvm::disasm::stats(&insns);
        if stats.insns > limits.max_insns as usize {
            diags.push(
                Diagnostic::error(format!(
                    "`{}` has {} instructions; the kernel loads programs of at most {} instructions",
                    program.name, stats.insns, limits.max_insns
                ))
                .with_code(codes::CODEGEN_LIMIT)
                .with_primary(
                    program.span,
                    format!("{} instructions, at most {}", stats.insns, limits.max_insns),
                )
                .with_help("split the program with `stage()`, or unroll fewer loops"),
            );
        }
        let frames = vigil_codegen_llvm::frames(&built.object, &section, &program.name);
        if frames.stack > limits.stack_bytes as usize {
            diags.push(
                Diagnostic::error(format!("`{}` uses too much stack", program.name))
                    .with_code(codes::CODEGEN_LIMIT)
                    .with_primary(
                        program.span,
                        format!("{} bytes, at most {}", frames.stack, limits.stack_bytes),
                    )
                    .with_note(
                        "the stack of the program and of the loop callbacks it calls add up, \
                         each frame rounded up to 32 bytes",
                    ),
            );
        }
        if frames.depth > limits.max_call_depth as usize {
            diags.push(
                Diagnostic::error(format!("`{}` nests calls too deeply", program.name))
                    .with_code(codes::CODEGEN_LIMIT)
                    .with_primary(
                        program.span,
                        format!(
                            "{} frames in a chain of calls, at most {}",
                            frames.depth, limits.max_call_depth
                        ),
                    )
                    .with_note("each loop run by `bpf_loop` is a callback with its own frame")
                    .with_help("nest fewer long loops, or mark inner loops `@unroll`"),
            );
        }
        code.push(insns);
    }
    // the stages of a program split by `stage()` are a chain of tail calls,
    // one before each later stage. Tail calls through the user's own
    // `ProgArray` maps are chosen at run time, so we cannot count them here.
    for (program, tail_calls) in vigil_codegen_llvm::stage_chains(&built.output.programs) {
        if tail_calls > limits.max_tail_calls {
            diags.push(
                Diagnostic::error(format!(
                    "`{}` has {} stages, {tail_calls} tail calls; the kernel follows at most {}",
                    program.name,
                    tail_calls + 1,
                    limits.max_tail_calls
                ))
                .with_code(codes::CODEGEN_LIMIT)
                .with_primary(program.span, format!("{} stages", tail_calls + 1)),
            );
        }
    }
    if diags.has_errors() {
        return None;
    }
    Some(Compiled {
        project,
        programs: built.output.programs,
        ir: built.output.ir,
        object: built.object,
        code,
    })
}

/// compiling a project with the certified backend, without LLVM. Only XDP
/// and TC programs of Vigil Core are accepted.
fn compile_certified(
    path: &Path,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<Compiled> {
    let (project, analysis) = analyze(path, sources, diags)?;
    let mut programs = Vec::new();
    for result in vigil_driver::certify(&analysis) {
        match result {
            Ok(program) => programs.push(program),
            Err(diag) => {
                // a map gives the same error for every program, once is enough
                let diag = diag.with_help("build without `--certified` (LLVM)");
                if !diags.iter().any(|d| *d == diag) {
                    diags.push(diag);
                }
            }
        }
    }
    if programs.is_empty() && !diags.has_errors() {
        diags.push(
            Diagnostic::error("this project has no programs")
                .with_code(codes::CODEGEN_LIMIT)
                .with_help("`--certified` builds XDP and TC programs of Vigil Core"),
        );
    }
    if diags.has_errors() {
        return None;
    }
    Some(Compiled {
        project,
        object: vigil_driver::certified_object(&programs),
        code: programs.iter().map(|p| p.code.clone()).collect(),
        programs: programs.into_iter().map(|p| p.desc).collect(),
        ir: String::new(),
    })
}

/// printing the success line of `vigil check`, and with `--verbose` the
/// profile, its features and limits
fn print_checked(project: &Project, verbose: bool) {
    let files = project.files.len();
    let plural = if files == 1 { "" } else { "s" };
    println!(
        "checked `{}` ({files} file{plural}) for {}",
        project.name, project.profile
    );
    if verbose {
        let profile = &project.profile;
        let limits = profile.limits();
        let features: Vec<_> = profile.features().map(|f| f.name()).collect();
        println!("  baseline:   {}", profile.baseline());
        println!("  arch:       {}", profile.arch());
        println!("  features:   {}", features.join(", "));
        println!("  insns:      {}", limits.max_insns);
        println!("  stack:      {} bytes", limits.stack_bytes);
        println!("  call depth: {}", limits.max_call_depth);
        println!("  tail calls: {}", limits.max_tail_calls);
        if project.libraries.std.is_none() {
            println!("  std:        built in");
        }
        for library in &project.libraries.resolved {
            println!("  {:<11} {}", format!("{}:", library.name), library.origin);
        }
    }
}

/// printing the libraries after `vigil update`
fn print_libraries(project: &Project) {
    if project.libraries.resolved.is_empty() {
        println!("`{}` has no dependencies", project.name);
    }
    for library in &project.libraries.resolved {
        println!("{:<12} {}", library.name, library.origin);
    }
}

/// writing the output to `output`, or by default to
/// `target/vigil/<name>.<ext>` under the project root (the current
/// directory for a lone file)
fn write_output(
    compiled: &Compiled,
    path: &Path,
    output: Option<&Path>,
    emit: Emit,
    diags: &mut Diagnostics,
) {
    let extension = match emit {
        Emit::Object => "o",
        Emit::Ir => "ll",
        Emit::Asm => "s",
    };
    let output = match output {
        Some(output) => output.to_path_buf(),
        None => match &compiled.project.root {
            Some(root) => root
                .join("target")
                .join("vigil")
                .join(format!("{}.{extension}", compiled.project.name)),
            None => PathBuf::from(format!("{}.{extension}", compiled.project.name)),
        },
    };
    let listing;
    let contents: &[u8] = match emit {
        Emit::Object => &compiled.object,
        // here the IR is never empty, because `build` rejects `--certified`
        // with `--emit ir`
        Emit::Ir => compiled.ir.as_bytes(),
        Emit::Asm => {
            listing = compiled
                .programs
                .iter()
                .zip(&compiled.code)
                .map(|(p, code)| {
                    format!("{}:\n{}", p.name, vigil_codegen_llvm::disasm::listing(code))
                })
                .collect::<Vec<_>>()
                .join("\n");
            listing.as_bytes()
        }
    };
    let written = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|_| std::fs::write(&output, contents));
    if let Err(err) = written {
        diags.push(
            Diagnostic::error(format!("could not write `{}`", output.display()))
                .with_code(codes::IO_ERROR)
                .with_note(err.to_string()),
        );
        return;
    }
    let programs: Vec<_> = compiled
        .programs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let name = format!("`{}` ({})", p.name, p.info.section());
            match compiled.code.get(i) {
                // the stack shown is the one checked against the limit
                Some(code) => {
                    let stats = vigil_codegen_llvm::disasm::stats(code);
                    let stack = vigil_codegen_llvm::stack_depth(
                        &compiled.object,
                        &p.info.section(),
                        &p.name,
                    );
                    format!("{name}, {} insns, {stack} B of stack", stats.insns)
                }
                None => name,
            }
        })
        .collect();
    // a project is named as `check` names it, a lone file by its path
    let built = match &compiled.project.root {
        Some(_) => compiled.project.name.clone(),
        None => path.display().to_string(),
    };
    println!(
        "built `{built}` → {} ({})",
        output.display(),
        programs.join(", ")
    );
}

/// choosing the item named `name`, or the only one when there is no name.
/// It is an error if none or several match.
fn choose<'a, T>(
    items: &'a [T],
    name: Option<&str>,
    name_of: impl Fn(&T) -> &str,
    diags: &mut Diagnostics,
) -> Option<&'a T> {
    let matching: Vec<_> = items
        .iter()
        .filter(|p| name.is_none_or(|n| name_of(p) == n))
        .collect();
    let names = |items: &mut dyn Iterator<Item = &T>| {
        items
            .map(|p| format!("`{}`", name_of(p)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match (matching.as_slice(), name) {
        ([one], _) => Some(one),
        // the same error as `build --certified` on a project without
        // programs
        (_, _) if items.is_empty() => {
            diags.push(
                Diagnostic::error("this project has no programs")
                    .with_code(codes::CODEGEN_LIMIT)
                    .with_help("these commands take an XDP or TC program of Vigil Core"),
            );
            None
        }
        ([], Some(n)) => {
            diags.push(
                Diagnostic::error(format!("there is no program `{n}`"))
                    .with_code(codes::INVALID_OPTIONS)
                    .with_note(format!("the programs are {}", names(&mut items.iter()))),
            );
            None
        }
        _ => {
            diags.push(
                Diagnostic::error(format!(
                    "this project has several programs: {}",
                    names(&mut matching.iter().copied())
                ))
                .with_code(codes::INVALID_OPTIONS)
                .with_help("choose one with `--program <NAME>`"),
            );
            None
        }
    }
}

/// printing a program as Vigil Core, readable or JSON
fn print_core(analysis: &Analysis, name: Option<&str>, json: bool, diags: &mut Diagnostics) {
    let programs = vigil_driver::programs(analysis);
    let Some(program) = choose(&programs, name, |p| &p.name, diags) else {
        return;
    };
    let Some(f) = analysis
        .programs
        .iter()
        .map(|(f, _)| f)
        .find(|f| analysis.package.fns[*f].name.text == program.name)
    else {
        return;
    };
    // the same errors as `vigil_driver::certify`, so `core`, `simulate` and
    // `build --certified` agree on what is in Vigil Core
    if let Some((_, map)) = analysis.package.maps.iter().next() {
        diags.push(
            Diagnostic::error("Vigil Core has no maps")
                .with_code(codes::CODEGEN_LIMIT)
                .with_primary(map.name.span, "a map keeps every program out of Vigil Core")
                .with_note(vigil_driver::CORE_SUBSET),
        );
        return;
    }
    match vigil_core::export::export(analysis, f) {
        Ok(core) if json => println!("{}", serde_json::to_string(&core).expect("Core serializes")),
        Ok(core) => print!("{}", vigil_core::print::print(&core)),
        Err(unsupported) => diags.push(
            Diagnostic::error(format!("`{}` is not in Vigil Core", program.name))
                .with_code(codes::CODEGEN_LIMIT)
                .with_primary(unsupported.span, unsupported.message)
                .with_note(vigil_driver::CORE_SUBSET),
        ),
    }
}

/// running a program on a packet in the interpreter of the driver
///
/// We run `model_code`, the code for the eBPF model of the formalization,
/// because the interpreter builds the model context, not the kernel one.
/// So only programs of the certified backend can be simulated.
fn simulate(analysis: &Analysis, name: Option<&str>, mut packet: Vec<u8>, diags: &mut Diagnostics) {
    let programs = vigil_driver::programs(analysis);
    let Some(program) = choose(&programs, name, |p| &p.name, diags) else {
        return;
    };
    let certified = vigil_driver::certify(analysis)
        .into_iter()
        .zip(&programs)
        .find(|(_, p)| p.name == program.name)
        .map(|(c, _)| c);
    let certified = match certified {
        Some(Ok(c)) => c,
        Some(Err(diag)) => {
            diags.push(diag);
            return;
        }
        None => return,
    };
    // the model code returns XDP verdicts, `verdict` gives the ones of
    // the program kind, as the playground does
    match vigil_driver::interp::run(&certified.model_code, &mut packet) {
        Ok(outcome) => {
            let (value, verdict) = vigil_driver::interp::verdict(program.info.kind, outcome.value);
            println!(
                "`{}` returned {value} ({verdict}) after {} instructions",
                program.name, outcome.steps
            )
        }
        Err(fault) => diags.push(
            Diagnostic::error(format!(
                "`{}` faulted in the eBPF interpreter",
                program.name
            ))
            .with_code(codes::SIMULATION_FAULT)
            .with_primary(program.span, "this program")
            .with_note(fault)
            .with_note(format!(
                "the certified backend should never produce a faulting program: \
                 unless the program ran out of steps, this is a bug in Vigil, \
                 please report it at {} with the program and the packet",
                vigil_diagnostics::ISSUES
            )),
        ),
    }
}

/// parsing the `--packet` of `vigil simulate`
fn parse_packet(text: &str, diags: &mut Diagnostics) -> Option<Vec<u8>> {
    match vigil_driver::interp::parse_hex(text) {
        Ok(packet) => Some(packet),
        Err(err) => {
            diags.push(
                Diagnostic::error(format!("invalid `--packet`: {err}"))
                    .with_code(codes::INVALID_PACKET)
                    .with_help(vigil_driver::interp::HEX_FORMAT),
            );
            None
        }
    }
}

/// parsing a file, adding its syntax errors to `diags`. `None` only when
/// the file cannot be read
fn parse_file(
    path: &Path,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<vigil_syntax::Parse> {
    let file = match sources.load(path) {
        Ok(file) => file,
        Err(err) => {
            diags.push(
                Diagnostic::error(format!("could not read `{}`", path.display()))
                    .with_code(codes::IO_ERROR)
                    .with_note(err.to_string()),
            );
            return None;
        }
    };
    let parse = vigil_syntax::parse(sources.get(file).text());
    for error in parse.errors() {
        diags.push(error.to_diagnostic(file));
    }
    Some(parse)
}

/// formatting the files under `paths` in place, or with `check` or `diff`
/// only showing what would change
///
/// For a project we take its sources, for another directory every `.vg`
/// file outside `target` directories. We fail on an unformatted file with
/// `check` or `diff`, and on a file we cannot read, parse or write. If
/// formatting changes the meaning of a file (a formatter bug), we leave it
/// alone.
fn format_files(paths: &[PathBuf], check: bool, diff: bool) -> ExitCode {
    let mut files = Vec::new();
    for path in paths {
        if path.join(vigil_target::MANIFEST_NAME).is_file() {
            files.extend(
                project::watched_files(path)
                    .into_iter()
                    .filter(|f| f.extension().is_some_and(|e| e == "vg")),
            );
        } else if path.is_dir() {
            let mut dirs = vec![path.clone()];
            while let Some(dir) = dirs.pop() {
                let Ok(entries) = std::fs::read_dir(&dir) else {
                    continue;
                };
                for entry in entries.flatten() {
                    let file = entry.path();
                    if file.is_dir() && !file.ends_with("target") {
                        dirs.push(file);
                    } else if file.extension().is_some_and(|e| e == "vg") {
                        files.push(file);
                    }
                }
            }
        } else {
            files.push(path.clone());
        }
    }
    files.sort();
    let mut status = ExitCode::SUCCESS;
    for file in files {
        let text = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("error: could not read `{}`: {err}", file.display());
                status = ExitCode::FAILURE;
                continue;
            }
        };
        let formatted = match vigil_syntax::fmt::format(&text) {
            Ok(formatted) => formatted,
            Err(vigil_syntax::fmt::Error::Syntax) => {
                eprintln!(
                    "error: `{}` has syntax errors (see `vigil check`)",
                    file.display()
                );
                status = ExitCode::FAILURE;
                continue;
            }
            Err(vigil_syntax::fmt::Error::Changed) => {
                eprintln!(
                    "error: formatting `{}` would change its meaning, so it is left as is; \
                     this is a bug in `vigil fmt`, please report it at {} with the file",
                    file.display(),
                    vigil_diagnostics::ISSUES
                );
                status = ExitCode::FAILURE;
                continue;
            }
        };
        if formatted == text {
            continue;
        }
        if check {
            println!("{} is not formatted", file.display());
            status = ExitCode::FAILURE;
        } else if diff {
            print!(
                "{}",
                diff::unified(&text, &formatted, &file.display().to_string())
            );
            status = ExitCode::FAILURE;
        } else if let Err(err) = std::fs::write(&file, formatted) {
            eprintln!("error: could not write `{}`: {err}", file.display());
            status = ExitCode::FAILURE;
        } else {
            println!("formatted {}", file.display());
        }
    }
    status
}

/// printing the diagnostics to stderr, failing if there is an error
fn report(sources: &SourceMap, diags: &Diagnostics, color: bool) -> ExitCode {
    let mut stderr = io::stderr().lock();
    let _ = Renderer::new(sources, color).render_all(diags, &mut stderr);
    let errors = diags.error_count();
    if errors > 0 {
        let plural = if errors == 1 { "" } else { "s" };
        let _ = writeln!(stderr, "aborting due to {errors} error{plural}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
