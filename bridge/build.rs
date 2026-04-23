fn main() {
    println!("cargo:rustc-link-search=native=/mnt/data/programming/KernelX/RadishDB");
    println!("cargo:rustc-link-lib=static=radish");
}
