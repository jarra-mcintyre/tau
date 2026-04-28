pub mod context;
pub mod providers;
pub mod tools;

/// Returns the human-readable name of the Tau tool.
pub fn name() -> &'static str {
    "Tau"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_name() {
        assert_eq!(name(), "Tau");
    }
}
