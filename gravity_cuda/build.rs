use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn find_tool(names: &[&str]) -> PathBuf {
    names
        .iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.is_file() || !candidate.components().count().gt(&1))
        .unwrap_or_else(|| panic!("none of these tools was found: {}", names.join(", ")))
}

fn find_nvcc() -> PathBuf {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("CUDA_NVCC") {
        candidates.push(PathBuf::from(path));
    }
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(variable) {
            let root = PathBuf::from(root);
            candidates.push(root.join("bin/nvcc"));
            candidates.push(root.join("bin/nvcc.exe"));
        }
    }
    candidates.extend([PathBuf::from("nvcc"), PathBuf::from("nvcc.exe")]);

    candidates
        .into_iter()
        .find(|candidate| {
            candidate == Path::new("nvcc")
                || candidate == Path::new("nvcc.exe")
                || candidate.is_file()
        })
        .unwrap_or_else(|| {
            panic!(
                "CUDA Toolkit was not found. Install nvcc or set CUDA_NVCC, CUDA_HOME, or CUDA_PATH."
            )
        })
}

fn run_checked(mut command: Command, description: &str) -> Vec<u8> {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to execute {description}: {error}"));
    if !output.status.success() {
        panic!(
            "{description} failed ({}):\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output.stdout
}

fn generate_windows_import_library(manifest_dir: &Path, out_dir: &Path) -> PathBuf {
    let dll = env::var_os("CUDA_DLL")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("nvcuda.dll"));
    println!("cargo:rerun-if-changed={}", dll.display());
    if !dll.is_file() {
        panic!(
            "CUDA driver DLL was not found at {}. Set CUDA_DLL to a Windows nvcuda.dll path, or place nvcuda.dll in the gravity_cuda project directory.",
            dll.display()
        );
    }

    let objdump = find_tool(&["llvm-objdump", "x86_64-w64-mingw32-objdump"]);
    let dump = run_checked(
        {
            let mut command = Command::new(objdump);
            command.args(["-p"]).arg(&dll);
            command
        },
        "CUDA DLL export inspection",
    );
    let dump_text = String::from_utf8_lossy(&dump);
    let exports: Vec<String> = dump_text
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let ordinal = fields.next()?;
            let address = fields.next()?;
            let name = fields.next()?;
            if ordinal.chars().all(|c| c.is_ascii_digit()) && address.starts_with("0x") {
                Some(name.to_owned())
            } else {
                None
            }
        })
        .filter(|name| !name.starts_with("#"))
        .collect();
    if exports.is_empty() {
        panic!("no exports were found in CUDA DLL {}", dll.display());
    }

    let def_path = out_dir.join("nvcuda.def");
    let mut definition = String::from("LIBRARY nvcuda.dll\nEXPORTS\n");
    for name in exports {
        definition.push_str("    ");
        definition.push_str(&name);
        definition.push('\n');
    }
    fs::write(&def_path, definition).expect("failed to write generated nvcuda.def");

    let dlltool = find_tool(&["x86_64-w64-mingw32-dlltool", "dlltool", "llvm-dlltool"]);
    let import_path = out_dir.join("libcuda.dll.a");
    run_checked(
        {
            let mut command = Command::new(dlltool);
            command
                .args(["--dllname", "nvcuda.dll", "--def"])
                .arg(&def_path)
                .arg("--output-lib")
                .arg(&import_path);
            command
        },
        "CUDA GNU import-library generation",
    );
    if !import_path.is_file() {
        panic!(
            "CUDA import library was not generated at {}",
            import_path.display()
        );
    }
    import_path
}

fn windows_import_library(target: &str, manifest_dir: &Path, out_dir: &Path) -> Option<PathBuf> {
    if target != "x86_64-pc-windows-gnu" {
        return None;
    }

    if let Some(explicit) = env::var_os("CUDA_IMPORT_LIB") {
        let library = PathBuf::from(explicit);
        if !library.is_file() {
            panic!(
                "CUDA_IMPORT_LIB does not point to a file: {}",
                library.display()
            );
        }
        println!("cargo:rerun-if-changed={}", library.display());
        return Some(library);
    }
    if let Some(directory) = env::var_os("CUDA_WINDOWS_LIB_DIR") {
        let library = PathBuf::from(directory).join("libcuda.dll.a");
        if !library.is_file() {
            panic!(
                "CUDA_WINDOWS_LIB_DIR does not contain libcuda.dll.a: {}",
                library.display()
            );
        }
        println!("cargo:rerun-if-changed={}", library.display());
        return Some(library);
    }
    Some(generate_windows_import_library(manifest_dir, out_dir))
}

fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let kernel = manifest_dir.join("kernels/gravity.cu");
    println!("cargo:rerun-if-changed={}", kernel.display());
    for variable in [
        "CUDA_NVCC",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ARCH",
        "CUDA_DLL",
        "CUDA_IMPORT_LIB",
        "CUDA_WINDOWS_LIB_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let target = env::var("TARGET").expect("TARGET is set by Cargo");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    if let Some(library) = windows_import_library(&target, &manifest_dir, &out_dir) {
        let directory = library
            .parent()
            .expect("CUDA import library has a parent directory");
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    let ptx_path = out_dir.join("gravity.ptx");
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_52".to_owned());
    if !arch.starts_with("compute_") && !arch.starts_with("sm_") {
        panic!("CUDA_ARCH must look like compute_XX or sm_XX, got {arch}");
    }
    run_checked(
        {
            let mut command = Command::new(find_nvcc());
            command
                .args(["--ptx", "-O3", "--use_fast_math", "-lineinfo"])
                .arg(format!("-arch={arch}"))
                .arg(&kernel)
                .arg("-o")
                .arg(&ptx_path);
            command
        },
        "nvcc PTX compilation",
    );

    println!("cargo:rustc-env=GRAVITY_CUDA_PTX={}", ptx_path.display());
}
