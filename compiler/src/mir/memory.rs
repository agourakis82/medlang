//! MIR Memory Model
//!
//! Defines the memory model and ownership tracking for MIR.
//! MedLang uses a region-based memory model with explicit ownership.

use std::collections::HashMap;

use super::types::*;
use super::value::*;

/// Memory region identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryRegion(pub u32);

impl MemoryRegion {
    /// Stack region (default for locals)
    pub const STACK: MemoryRegion = MemoryRegion(0);
    /// Heap region (boxed allocations)
    pub const HEAP: MemoryRegion = MemoryRegion(1);
    /// Arena region (bulk allocations)
    pub const ARENA: MemoryRegion = MemoryRegion(2);
    /// Static region (globals and constants)
    pub const STATIC: MemoryRegion = MemoryRegion(3);
    /// GPU device memory
    pub const GPU: MemoryRegion = MemoryRegion(4);
    /// Pinned memory (for GPU transfers)
    pub const PINNED: MemoryRegion = MemoryRegion(5);

    pub fn new(id: u32) -> Self {
        MemoryRegion(id)
    }

    /// Is this a heap-like region (requires deallocation)?
    pub fn is_heap(&self) -> bool {
        matches!(
            *self,
            MemoryRegion::HEAP | MemoryRegion::ARENA | MemoryRegion::GPU | MemoryRegion::PINNED
        )
    }

    /// Is this a device memory region?
    pub fn is_device(&self) -> bool {
        matches!(*self, MemoryRegion::GPU)
    }
}

impl std::fmt::Display for MemoryRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            MemoryRegion::STACK => write!(f, "stack"),
            MemoryRegion::HEAP => write!(f, "heap"),
            MemoryRegion::ARENA => write!(f, "arena"),
            MemoryRegion::STATIC => write!(f, "static"),
            MemoryRegion::GPU => write!(f, "gpu"),
            MemoryRegion::PINNED => write!(f, "pinned"),
            MemoryRegion(id) => write!(f, "region{}", id),
        }
    }
}

/// Memory location
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemoryLocation {
    /// Base pointer value
    pub base: ValueId,
    /// Offset from base
    pub offset: usize,
    /// Size of access
    pub size: usize,
    /// Memory region
    pub region: MemoryRegion,
}

impl MemoryLocation {
    pub fn new(base: ValueId, region: MemoryRegion) -> Self {
        Self {
            base,
            offset: 0,
            size: 0,
            region,
        }
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    pub fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Check if two locations may alias
    pub fn may_alias(&self, other: &MemoryLocation) -> bool {
        // Different regions cannot alias
        if self.region != other.region {
            return false;
        }

        // Same base, check for overlap
        if self.base == other.base {
            let self_end = self.offset + self.size;
            let other_end = other.offset + other.size;

            // No overlap if one ends before the other starts
            if self_end <= other.offset || other_end <= self.offset {
                return false;
            }
            return true;
        }

        // Different bases, conservatively assume they may alias
        true
    }

    /// Check if this location must alias another
    pub fn must_alias(&self, other: &MemoryLocation) -> bool {
        self.region == other.region
            && self.base == other.base
            && self.offset == other.offset
            && self.size == other.size
    }
}

/// Ownership kind
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ownership {
    /// Owned (responsible for deallocation)
    Owned,
    /// Borrowed immutably
    Borrowed,
    /// Borrowed mutably (exclusive)
    BorrowedMut,
    /// Shared (reference counted internally)
    Shared,
    /// Raw (no ownership tracking)
    Raw,
}

impl Ownership {
    /// Can this ownership be read from?
    pub fn can_read(&self) -> bool {
        matches!(
            self,
            Ownership::Owned
                | Ownership::Borrowed
                | Ownership::BorrowedMut
                | Ownership::Shared
                | Ownership::Raw
        )
    }

    /// Can this ownership be written to?
    pub fn can_write(&self) -> bool {
        matches!(
            self,
            Ownership::Owned | Ownership::BorrowedMut | Ownership::Raw
        )
    }

    /// Does this ownership require deallocation?
    pub fn requires_drop(&self) -> bool {
        matches!(self, Ownership::Owned | Ownership::Shared)
    }
}

/// Pointer provenance (for alias analysis)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Provenance {
    /// Allocation site ID
    pub allocation: AllocationId,
    /// Access path from allocation
    pub path: Vec<AccessPathElement>,
}

impl Provenance {
    pub fn new(allocation: AllocationId) -> Self {
        Self {
            allocation,
            path: Vec::new(),
        }
    }

    pub fn with_field(mut self, field: usize) -> Self {
        self.path.push(AccessPathElement::Field(field));
        self
    }

    pub fn with_index(mut self, index: usize) -> Self {
        self.path.push(AccessPathElement::Index(index));
        self
    }

    pub fn with_deref(mut self) -> Self {
        self.path.push(AccessPathElement::Deref);
        self
    }
}

/// Access path element
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessPathElement {
    /// Field access
    Field(usize),
    /// Array/slice index
    Index(usize),
    /// Pointer dereference
    Deref,
}

/// Allocation identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationId(pub u32);

impl AllocationId {
    pub fn new(id: u32) -> Self {
        AllocationId(id)
    }
}

impl std::fmt::Display for AllocationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "alloc{}", self.0)
    }
}

/// Allocation info
#[derive(Clone, Debug)]
pub struct Allocation {
    /// Allocation ID
    pub id: AllocationId,
    /// Type allocated
    pub ty: MirType,
    /// Region where allocated
    pub region: MemoryRegion,
    /// Size in bytes
    pub size: usize,
    /// Alignment
    pub align: usize,
    /// Is this allocation live?
    pub live: bool,
    /// Allocation site (value that created it)
    pub site: Option<ValueId>,
}

impl Allocation {
    pub fn new(id: AllocationId, ty: MirType, region: MemoryRegion) -> Self {
        let size = ty.size();
        let align = ty.align();
        Self {
            id,
            ty,
            region,
            size,
            align,
            live: true,
            site: None,
        }
    }

    pub fn with_site(mut self, site: ValueId) -> Self {
        self.site = Some(site);
        self
    }

    pub fn kill(&mut self) {
        self.live = false;
    }
}

/// Memory state tracker
#[derive(Debug, Default)]
pub struct MemoryState {
    /// Allocations
    allocations: HashMap<AllocationId, Allocation>,
    /// Next allocation ID
    next_alloc_id: u32,
    /// Value to allocation mapping
    value_provenance: HashMap<ValueId, Provenance>,
    /// Value ownership
    value_ownership: HashMap<ValueId, Ownership>,
}

impl MemoryState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate memory
    pub fn allocate(&mut self, ty: MirType, region: MemoryRegion) -> AllocationId {
        let id = AllocationId(self.next_alloc_id);
        self.next_alloc_id += 1;

        let alloc = Allocation::new(id, ty, region);
        self.allocations.insert(id, alloc);
        id
    }

    /// Deallocate memory
    pub fn deallocate(&mut self, id: AllocationId) -> Result<(), MemoryError> {
        if let Some(alloc) = self.allocations.get_mut(&id) {
            if !alloc.live {
                return Err(MemoryError::DoubleFree(id));
            }
            alloc.kill();
            Ok(())
        } else {
            Err(MemoryError::InvalidAllocation(id))
        }
    }

    /// Set provenance for a value
    pub fn set_provenance(&mut self, value: ValueId, provenance: Provenance) {
        self.value_provenance.insert(value, provenance);
    }

    /// Get provenance for a value
    pub fn get_provenance(&self, value: ValueId) -> Option<&Provenance> {
        self.value_provenance.get(&value)
    }

    /// Set ownership for a value
    pub fn set_ownership(&mut self, value: ValueId, ownership: Ownership) {
        self.value_ownership.insert(value, ownership);
    }

    /// Get ownership for a value
    pub fn get_ownership(&self, value: ValueId) -> Option<Ownership> {
        self.value_ownership.get(&value).copied()
    }

    /// Check if a value can be read
    pub fn can_read(&self, value: ValueId) -> bool {
        self.value_ownership
            .get(&value)
            .map(|o| o.can_read())
            .unwrap_or(true)
    }

    /// Check if a value can be written
    pub fn can_write(&self, value: ValueId) -> bool {
        self.value_ownership
            .get(&value)
            .map(|o| o.can_write())
            .unwrap_or(true)
    }

    /// Get allocation by ID
    pub fn get_allocation(&self, id: AllocationId) -> Option<&Allocation> {
        self.allocations.get(&id)
    }

    /// Check if allocation is live
    pub fn is_live(&self, id: AllocationId) -> bool {
        self.allocations.get(&id).map(|a| a.live).unwrap_or(false)
    }

    /// Get all live allocations
    pub fn live_allocations(&self) -> impl Iterator<Item = &Allocation> {
        self.allocations.values().filter(|a| a.live)
    }

    /// Check for memory leaks
    pub fn check_leaks(&self) -> Vec<AllocationId> {
        self.allocations
            .values()
            .filter(|a| a.live && a.region.is_heap())
            .map(|a| a.id)
            .collect()
    }
}

/// Memory error
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryError {
    /// Double free
    DoubleFree(AllocationId),
    /// Use after free
    UseAfterFree(AllocationId),
    /// Invalid allocation
    InvalidAllocation(AllocationId),
    /// Aliasing violation
    AliasingViolation { existing: ValueId, new: ValueId },
    /// Write to borrowed value
    WriteToBorrowed(ValueId),
    /// Read from moved value
    ReadFromMoved(ValueId),
    /// Memory leak
    Leak(AllocationId),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::DoubleFree(id) => write!(f, "double free of {}", id),
            MemoryError::UseAfterFree(id) => write!(f, "use after free of {}", id),
            MemoryError::InvalidAllocation(id) => write!(f, "invalid allocation {}", id),
            MemoryError::AliasingViolation { existing, new } => {
                write!(f, "aliasing violation: {} conflicts with {}", new, existing)
            }
            MemoryError::WriteToBorrowed(id) => {
                write!(f, "write to immutably borrowed value {}", id)
            }
            MemoryError::ReadFromMoved(id) => write!(f, "read from moved value {}", id),
            MemoryError::Leak(id) => write!(f, "memory leak: {} not freed", id),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Alias analysis result
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AliasResult {
    /// No aliasing possible
    NoAlias,
    /// May alias
    MayAlias,
    /// Must alias (same location)
    MustAlias,
    /// Partial overlap
    PartialAlias,
}

/// Basic alias analyzer
pub struct AliasAnalyzer {
    memory_state: MemoryState,
}

impl AliasAnalyzer {
    pub fn new(memory_state: MemoryState) -> Self {
        Self { memory_state }
    }

    /// Check if two values may alias
    pub fn alias(&self, a: ValueId, b: ValueId) -> AliasResult {
        // Same value must alias
        if a == b {
            return AliasResult::MustAlias;
        }

        let prov_a = self.memory_state.get_provenance(a);
        let prov_b = self.memory_state.get_provenance(b);

        match (prov_a, prov_b) {
            (Some(pa), Some(pb)) => {
                // Different allocations cannot alias
                if pa.allocation != pb.allocation {
                    return AliasResult::NoAlias;
                }

                // Same allocation, check paths
                if pa.path == pb.path {
                    return AliasResult::MustAlias;
                }

                // Check for prefix relationship
                if Self::is_prefix(&pa.path, &pb.path) || Self::is_prefix(&pb.path, &pa.path) {
                    return AliasResult::PartialAlias;
                }

                // Different paths, may alias through arrays
                if Self::paths_may_overlap(&pa.path, &pb.path) {
                    return AliasResult::MayAlias;
                }

                AliasResult::NoAlias
            }
            // Without provenance, conservatively assume may alias
            _ => AliasResult::MayAlias,
        }
    }

    fn is_prefix(shorter: &[AccessPathElement], longer: &[AccessPathElement]) -> bool {
        if shorter.len() > longer.len() {
            return false;
        }
        shorter.iter().zip(longer.iter()).all(|(a, b)| a == b)
    }

    fn paths_may_overlap(a: &[AccessPathElement], b: &[AccessPathElement]) -> bool {
        // If either path contains an index, they may overlap
        a.iter()
            .chain(b.iter())
            .any(|e| matches!(e, AccessPathElement::Index(_)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_region() {
        assert!(MemoryRegion::HEAP.is_heap());
        assert!(MemoryRegion::GPU.is_device());
        assert!(!MemoryRegion::STACK.is_heap());
        assert!(!MemoryRegion::STATIC.is_device());
    }

    #[test]
    fn test_memory_location_aliasing() {
        let loc1 = MemoryLocation::new(ValueId(0), MemoryRegion::STACK)
            .with_offset(0)
            .with_size(8);
        let loc2 = MemoryLocation::new(ValueId(0), MemoryRegion::STACK)
            .with_offset(8)
            .with_size(8);
        let loc3 = MemoryLocation::new(ValueId(0), MemoryRegion::STACK)
            .with_offset(4)
            .with_size(8);

        // Non-overlapping
        assert!(!loc1.may_alias(&loc2));

        // Overlapping
        assert!(loc1.may_alias(&loc3));

        // Different regions
        let loc4 = MemoryLocation::new(ValueId(0), MemoryRegion::HEAP)
            .with_offset(0)
            .with_size(8);
        assert!(!loc1.may_alias(&loc4));
    }

    #[test]
    fn test_ownership() {
        assert!(Ownership::Owned.can_read());
        assert!(Ownership::Owned.can_write());
        assert!(Ownership::Borrowed.can_read());
        assert!(!Ownership::Borrowed.can_write());
        assert!(Ownership::BorrowedMut.can_write());
        assert!(Ownership::Owned.requires_drop());
        assert!(!Ownership::Borrowed.requires_drop());
    }

    #[test]
    fn test_memory_state() {
        let mut state = MemoryState::new();

        let alloc1 = state.allocate(MirType::F64, MemoryRegion::HEAP);
        let alloc2 = state.allocate(MirType::I32, MemoryRegion::STACK);

        assert!(state.is_live(alloc1));
        assert!(state.is_live(alloc2));

        assert!(state.deallocate(alloc1).is_ok());
        assert!(!state.is_live(alloc1));

        // Double free should error
        assert!(matches!(
            state.deallocate(alloc1),
            Err(MemoryError::DoubleFree(_))
        ));
    }

    #[test]
    fn test_provenance() {
        let prov = Provenance::new(AllocationId(0))
            .with_field(0)
            .with_index(1)
            .with_deref();

        assert_eq!(prov.path.len(), 3);
        assert_eq!(prov.path[0], AccessPathElement::Field(0));
        assert_eq!(prov.path[1], AccessPathElement::Index(1));
        assert_eq!(prov.path[2], AccessPathElement::Deref);
    }

    #[test]
    fn test_alias_analyzer() {
        let mut state = MemoryState::new();

        let alloc1 = state.allocate(MirType::array(MirType::F64, 10), MemoryRegion::HEAP);
        let alloc2 = state.allocate(MirType::F64, MemoryRegion::HEAP);

        // Same allocation, different fields - NoAlias
        state.set_provenance(ValueId(0), Provenance::new(alloc1).with_field(0));
        state.set_provenance(ValueId(1), Provenance::new(alloc1).with_field(1));

        let analyzer = AliasAnalyzer::new(state);

        // Same value must alias
        assert_eq!(
            analyzer.alias(ValueId(0), ValueId(0)),
            AliasResult::MustAlias
        );

        // Different allocations cannot alias
        let mut state2 = MemoryState::new();
        state2.set_provenance(ValueId(0), Provenance::new(alloc1));
        state2.set_provenance(ValueId(1), Provenance::new(alloc2));
        let analyzer2 = AliasAnalyzer::new(state2);
        assert_eq!(
            analyzer2.alias(ValueId(0), ValueId(1)),
            AliasResult::NoAlias
        );
    }

    #[test]
    fn test_leak_detection() {
        let mut state = MemoryState::new();

        let alloc1 = state.allocate(MirType::F64, MemoryRegion::HEAP);
        let alloc2 = state.allocate(MirType::F64, MemoryRegion::STACK);
        let _alloc3 = state.allocate(MirType::F64, MemoryRegion::HEAP);

        // Free first heap allocation
        state.deallocate(alloc1).unwrap();

        // Stack doesn't count as leak
        // First heap is freed
        // Second heap is a leak
        let leaks = state.check_leaks();
        assert_eq!(leaks.len(), 1);
        assert!(leaks.iter().all(|id| *id != alloc1 && *id != alloc2));
    }
}
