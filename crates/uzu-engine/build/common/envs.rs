use std::sync::OnceLock;

fn env_flag(name: &str) -> bool {
    println!("cargo::rerun-if-env-changed={name}");
    std::env::var(name).is_ok()
}

pub fn build_debug() -> bool {
    static BUILD_DEBUG: OnceLock<bool> = OnceLock::new();
    *BUILD_DEBUG.get_or_init(|| env_flag("BUILD_DEBUG"))
}

pub fn build_always() -> bool {
    static BUILD_ALWAYS: OnceLock<bool> = OnceLock::new();
    *BUILD_ALWAYS.get_or_init(|| env_flag("BUILD_ALWAYS"))
}

pub fn build_clean() -> bool {
    static BUILD_CLEAN: OnceLock<bool> = OnceLock::new();
    *BUILD_CLEAN.get_or_init(|| env_flag("BUILD_CLEAN"))
}
