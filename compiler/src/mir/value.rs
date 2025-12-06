//! MIR Values and Identifiers
//!
//! Defines the value and block identifiers used in SSA form MIR.

use std::fmt;

/// Value identifier in SSA form
///
/// Each value in MIR has a unique ID. Values are assigned once and
/// never modified (SSA property).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl ValueId {
    /// Create a new value ID
    pub fn new(id: u32) -> Self {
        ValueId(id)
    }

    /// Get the raw ID value
    pub fn id(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl From<u32> for ValueId {
    fn from(id: u32) -> Self {
        ValueId(id)
    }
}

impl From<usize> for ValueId {
    fn from(id: usize) -> Self {
        ValueId(id as u32)
    }
}

/// Basic block identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Entry block ID (always 0)
    pub const ENTRY: BlockId = BlockId(0);

    /// Create a new block ID
    pub fn new(id: u32) -> Self {
        BlockId(id)
    }

    /// Get the raw ID value
    pub fn id(&self) -> u32 {
        self.0
    }

    /// Check if this is the entry block
    pub fn is_entry(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl From<u32> for BlockId {
    fn from(id: u32) -> Self {
        BlockId(id)
    }
}

impl From<usize> for BlockId {
    fn from(id: usize) -> Self {
        BlockId(id as u32)
    }
}

/// Generator for unique value IDs
#[derive(Debug, Default)]
pub struct ValueIdGen {
    next_id: u32,
}

impl ValueIdGen {
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    /// Generate a new unique value ID
    pub fn next(&mut self) -> ValueId {
        let id = ValueId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Get the current count of generated IDs
    pub fn count(&self) -> u32 {
        self.next_id
    }

    /// Reset the generator (useful for testing)
    pub fn reset(&mut self) {
        self.next_id = 0;
    }
}

/// Generator for unique block IDs
#[derive(Debug, Default)]
pub struct BlockIdGen {
    next_id: u32,
}

impl BlockIdGen {
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    /// Generate a new unique block ID
    pub fn next(&mut self) -> BlockId {
        let id = BlockId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Get the current count of generated IDs
    pub fn count(&self) -> u32 {
        self.next_id
    }

    /// Reset the generator (useful for testing)
    pub fn reset(&mut self) {
        self.next_id = 0;
    }
}

/// Source location span
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
    pub file_id: u32,
}

impl Span {
    pub fn new(start: u32, end: u32, file_id: u32) -> Self {
        Self {
            start,
            end,
            file_id,
        }
    }

    /// Create a dummy span for generated code
    pub fn dummy() -> Self {
        Self {
            start: 0,
            end: 0,
            file_id: 0,
        }
    }

    /// Merge two spans into one covering both
    pub fn merge(self, other: Span) -> Span {
        if self.file_id != other.file_id {
            return self;
        }
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            file_id: self.file_id,
        }
    }

    /// Length of the span in bytes
    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// Check if span is empty
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}..{}", self.file_id, self.start, self.end)
    }
}

/// Scope identifier for debug info
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeId(pub u32);

impl ScopeId {
    /// Root scope
    pub const ROOT: ScopeId = ScopeId(0);

    pub fn new(id: u32) -> Self {
        ScopeId(id)
    }
}

impl fmt::Display for ScopeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "scope{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_id_gen() {
        let mut gen = ValueIdGen::new();
        assert_eq!(gen.next(), ValueId(0));
        assert_eq!(gen.next(), ValueId(1));
        assert_eq!(gen.next(), ValueId(2));
        assert_eq!(gen.count(), 3);
    }

    #[test]
    fn test_block_id_gen() {
        let mut gen = BlockIdGen::new();
        assert_eq!(gen.next(), BlockId(0));
        assert!(gen.next().id() == 1);
        assert_eq!(gen.count(), 2);
    }

    #[test]
    fn test_value_id_display() {
        assert_eq!(ValueId(42).to_string(), "v42");
    }

    #[test]
    fn test_block_id_display() {
        assert_eq!(BlockId(3).to_string(), "bb3");
        assert!(BlockId::ENTRY.is_entry());
    }

    #[test]
    fn test_span() {
        let span1 = Span::new(10, 20, 1);
        let span2 = Span::new(15, 30, 1);
        let merged = span1.merge(span2);

        assert_eq!(merged.start, 10);
        assert_eq!(merged.end, 30);
        assert_eq!(span1.len(), 10);
    }
}
