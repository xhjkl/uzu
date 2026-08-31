use std::{sync::LazyLock, time::Instant};

use crate::common::envs;

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

pub fn _debug_log(args: std::fmt::Arguments) {
    if envs::build_debug() {
        let elapsed_ms = START.elapsed().as_millis();
        println!("cargo::warning=(build-debug) [{elapsed_ms}ms] {args}");
    }
}

#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {{
        $crate::common::logging::_debug_log(format_args!($($arg)*));
    }};
}
