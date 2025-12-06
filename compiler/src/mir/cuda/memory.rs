//! CUDA Memory Planning and Optimization
//!
//! Plans shared memory allocation, memory coalescing, and bank conflict avoidance.

use std::collections::HashMap;

use super::types::{AddressSpace, CudaType, GpuArch};
use crate::mir::value::ValueId;

/// Shared memory allocation
#[derive(Clone, Debug)]
pub struct SharedMemoryAlloc {
    /// Variable name
    pub name: String,
    /// Type of element
    pub element_type: CudaType,
    /// Total size in elements
    pub num_elements: usize,
    /// Offset in shared memory (bytes)
    pub offset: usize,
    /// Padding for bank conflict avoidance
    pub padding: usize,
    /// Whether this is dynamically sized
    pub is_dynamic: bool,
}

impl SharedMemoryAlloc {
    /// Total size including padding
    pub fn total_bytes(&self) -> usize {
        (self.num_elements * self.element_type.size_bytes()) + self.padding
    }

    /// Generate CUDA declaration
    pub fn to_cuda_decl(&self) -> String {
        if self.is_dynamic {
            format!(
                "extern __shared__ {} {}[];",
                self.element_type.cuda_name(),
                self.name
            )
        } else if self.padding > 0 {
            // Padded array for bank conflict avoidance
            let padded_width = self.element_type.size_bytes() + self.padding / self.num_elements;
            format!(
                "__shared__ {} {}[{}]; // padded stride: {}",
                self.element_type.cuda_name(),
                self.name,
                self.num_elements,
                padded_width
            )
        } else {
            format!(
                "__shared__ {} {}[{}];",
                self.element_type.cuda_name(),
                self.name,
                self.num_elements
            )
        }
    }
}

/// Memory access pattern
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access (thread i accesses element i)
    Sequential,
    /// Strided access
    Strided { stride: usize },
    /// Random access
    Random,
    /// Broadcast (all threads read same element)
    Broadcast,
    /// Coalesced 2D access
    Coalesced2D { row_stride: usize },
}

/// Memory coalescing analysis result
#[derive(Clone, Debug)]
pub struct CoalescingAnalysis {
    /// Access pattern
    pub pattern: AccessPattern,
    /// Is the access coalesced?
    pub is_coalesced: bool,
    /// Number of memory transactions per warp
    pub transactions_per_warp: u32,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

impl CoalescingAnalysis {
    pub fn coalesced() -> Self {
        Self {
            pattern: AccessPattern::Sequential,
            is_coalesced: true,
            transactions_per_warp: 1,
            suggestions: Vec::new(),
        }
    }

    pub fn uncoalesced(pattern: AccessPattern, transactions: u32) -> Self {
        Self {
            pattern,
            is_coalesced: false,
            transactions_per_warp: transactions,
            suggestions: vec![
                "Consider reorganizing data layout".to_string(),
                "Use shared memory for data reuse".to_string(),
            ],
        }
    }
}

/// Shared memory bank conflict analysis
#[derive(Clone, Debug)]
pub struct BankConflictAnalysis {
    /// Number of banks (typically 32)
    pub num_banks: u32,
    /// Number of way conflicts
    pub way_conflicts: u32,
    /// Bank access pattern
    pub bank_pattern: Vec<u32>,
    /// Is conflict-free?
    pub is_conflict_free: bool,
}

impl BankConflictAnalysis {
    pub fn conflict_free() -> Self {
        Self {
            num_banks: 32,
            way_conflicts: 1,
            bank_pattern: Vec::new(),
            is_conflict_free: true,
        }
    }

    pub fn with_conflicts(way_conflicts: u32) -> Self {
        Self {
            num_banks: 32,
            way_conflicts,
            bank_pattern: Vec::new(),
            is_conflict_free: way_conflicts <= 1,
        }
    }

    /// Compute padding needed to avoid conflicts
    pub fn suggested_padding(&self, element_size: usize) -> usize {
        if self.is_conflict_free {
            return 0;
        }
        // Add one element of padding per row to shift bank access
        element_size
    }
}

/// Memory plan for a kernel
#[derive(Clone, Debug)]
pub struct MemoryPlan {
    /// Shared memory allocations
    pub shared_allocations: Vec<SharedMemoryAlloc>,
    /// Total static shared memory
    pub total_static_shared: usize,
    /// Dynamic shared memory requirement
    pub dynamic_shared: usize,
    /// Register estimate per thread
    pub registers_per_thread: u32,
    /// Memory access analyses
    pub access_analyses: HashMap<String, CoalescingAnalysis>,
    /// Bank conflict analyses
    pub bank_analyses: HashMap<String, BankConflictAnalysis>,
    /// Local memory spills (bytes per thread)
    pub local_memory_spills: usize,
}

impl MemoryPlan {
    pub fn new() -> Self {
        Self {
            shared_allocations: Vec::new(),
            total_static_shared: 0,
            dynamic_shared: 0,
            registers_per_thread: 32,
            access_analyses: HashMap::new(),
            bank_analyses: HashMap::new(),
            local_memory_spills: 0,
        }
    }

    /// Add a shared memory allocation
    pub fn add_shared_alloc(&mut self, alloc: SharedMemoryAlloc) {
        if !alloc.is_dynamic {
            self.total_static_shared += alloc.total_bytes();
        }
        self.shared_allocations.push(alloc);
    }

    /// Check if plan fits in available shared memory
    pub fn fits_shared_memory(&self, arch: GpuArch) -> bool {
        let max_shared = arch.max_shared_memory_per_block() as usize;
        self.total_static_shared + self.dynamic_shared <= max_shared
    }

    /// Estimate occupancy
    pub fn estimate_occupancy(&self, arch: GpuArch, threads_per_block: u32) -> f64 {
        let max_threads = arch.max_threads_per_block();
        let max_shared = arch.max_shared_memory_per_block() as usize;
        let max_regs = arch.max_registers_per_thread();

        // Thread limit
        let thread_limit = max_threads / threads_per_block;

        // Shared memory limit
        let shared_limit = if self.total_static_shared > 0 {
            max_shared / self.total_static_shared
        } else {
            u32::MAX as usize
        } as u32;

        // Register limit (simplified)
        // Total registers per SM is typically 65536, divided among all threads
        let reg_limit = if self.registers_per_thread > 0 {
            let total_regs_per_sm = 65536u32; // Standard for modern GPUs
            let regs_per_block = self.registers_per_thread * threads_per_block;
            if regs_per_block > 0 {
                total_regs_per_sm / regs_per_block
            } else {
                u32::MAX
            }
        } else {
            u32::MAX
        };

        let blocks_per_sm = thread_limit.min(shared_limit).min(reg_limit);
        let active_threads = blocks_per_sm * threads_per_block;

        // Theoretical max threads per SM varies by arch
        let max_threads_per_sm = match arch {
            GpuArch::Sm70 | GpuArch::Sm75 => 2048,
            GpuArch::Sm80 | GpuArch::Sm86 | GpuArch::Sm89 => 2048,
            GpuArch::Sm90 => 2048,
        };

        active_threads as f64 / max_threads_per_sm as f64
    }

    /// Generate all shared memory declarations
    pub fn generate_shared_decls(&self) -> String {
        self.shared_allocations
            .iter()
            .map(|a| a.to_cuda_decl())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for MemoryPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory planner for kernel optimization
pub struct MemoryPlanner {
    arch: GpuArch,
}

impl MemoryPlanner {
    pub fn new(arch: GpuArch) -> Self {
        Self { arch }
    }

    /// Plan shared memory for matrix multiplication tile
    pub fn plan_matmul_tiles(
        &self,
        m_tile: usize,
        n_tile: usize,
        k_tile: usize,
        element_type: CudaType,
    ) -> MemoryPlan {
        let mut plan = MemoryPlan::new();
        let elem_size = element_type.size_bytes();

        // Tile A: m_tile x k_tile
        let a_elements = m_tile * k_tile;
        // Add padding to avoid bank conflicts (pad by 1 element per row)
        let a_padding = m_tile * elem_size;
        plan.add_shared_alloc(SharedMemoryAlloc {
            name: "smem_A".to_string(),
            element_type: element_type.clone(),
            num_elements: a_elements,
            offset: 0,
            padding: a_padding,
            is_dynamic: false,
        });

        // Tile B: k_tile x n_tile
        let b_elements = k_tile * n_tile;
        let b_padding = k_tile * elem_size;
        plan.add_shared_alloc(SharedMemoryAlloc {
            name: "smem_B".to_string(),
            element_type,
            num_elements: b_elements,
            offset: a_elements * elem_size + a_padding,
            padding: b_padding,
            is_dynamic: false,
        });

        plan
    }

    /// Plan shared memory for reduction
    pub fn plan_reduction(&self, threads_per_block: usize, element_type: CudaType) -> MemoryPlan {
        let mut plan = MemoryPlan::new();

        plan.add_shared_alloc(SharedMemoryAlloc {
            name: "smem_reduce".to_string(),
            element_type,
            num_elements: threads_per_block,
            offset: 0,
            padding: 0, // No bank conflicts in reduction pattern
            is_dynamic: false,
        });

        plan
    }

    /// Analyze memory coalescing for array access
    pub fn analyze_coalescing(&self, stride: usize, element_size: usize) -> CoalescingAnalysis {
        let warp_size = self.arch.warp_size() as usize;

        if stride == element_size {
            // Perfect coalescing
            CoalescingAnalysis::coalesced()
        } else if stride == 0 {
            // Broadcast
            CoalescingAnalysis {
                pattern: AccessPattern::Broadcast,
                is_coalesced: true,
                transactions_per_warp: 1,
                suggestions: Vec::new(),
            }
        } else {
            // Strided access
            let transactions = ((warp_size * stride) / 128).max(1) as u32;
            CoalescingAnalysis::uncoalesced(AccessPattern::Strided { stride }, transactions)
        }
    }

    /// Analyze shared memory bank conflicts
    pub fn analyze_bank_conflicts(
        &self,
        stride: usize,
        element_size: usize,
    ) -> BankConflictAnalysis {
        let bank_size = 4; // 4 bytes per bank
        let num_banks = 32;

        // Compute effective stride in banks
        let stride_banks = (stride * element_size) / bank_size;

        if stride_banks % num_banks == 0 {
            // All threads hit same bank - worst case
            BankConflictAnalysis::with_conflicts(32)
        } else if stride_banks % 2 == 0 && stride_banks % num_banks != 0 {
            // 2-way conflicts possible
            let gcd = gcd(stride_banks, num_banks);
            BankConflictAnalysis::with_conflicts(gcd as u32)
        } else {
            // Conflict free
            BankConflictAnalysis::conflict_free()
        }
    }

    /// Compute optimal tile sizes for available shared memory
    pub fn compute_tile_sizes(
        &self,
        m: usize,
        n: usize,
        k: usize,
        element_size: usize,
        max_shared: usize,
    ) -> (usize, usize, usize) {
        // Start with desired tile sizes
        let mut m_tile = 128;
        let mut n_tile = 128;
        let mut k_tile = 32;

        loop {
            // Compute required shared memory
            // A tile: m_tile x k_tile, B tile: k_tile x n_tile
            let shared_required = (m_tile * k_tile + k_tile * n_tile) * element_size;

            if shared_required <= max_shared {
                break;
            }

            // Reduce tile sizes
            if k_tile > 8 {
                k_tile /= 2;
            } else if m_tile > 32 {
                m_tile /= 2;
            } else if n_tile > 32 {
                n_tile /= 2;
            } else {
                break; // Can't reduce further
            }
        }

        (m_tile, n_tile, k_tile)
    }
}

/// Greatest common divisor
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Data layout transformation for memory optimization
#[derive(Clone, Debug)]
pub enum DataLayoutTransform {
    /// No transformation needed
    None,
    /// Transpose matrix
    Transpose,
    /// Convert Array of Structures to Structure of Arrays
    AosToSoa,
    /// Convert Structure of Arrays to Array of Structures
    SoaToAos,
    /// Add padding for alignment
    Pad { amount: usize },
    /// Tile data for cache efficiency
    Tile { tile_size: Vec<usize> },
}

/// Memory prefetch hint
#[derive(Clone, Debug)]
pub struct PrefetchHint {
    /// Address to prefetch
    pub address: String,
    /// Cache level (L1, L2)
    pub cache_level: CacheLevel,
    /// Read or write
    pub access_type: PrefetchAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLevel {
    L1,
    L2,
    // L3 not applicable for GPU
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefetchAccess {
    Read,
    Write,
}

impl PrefetchHint {
    pub fn to_cuda(&self) -> String {
        let level = match self.cache_level {
            CacheLevel::L1 => ".L1",
            CacheLevel::L2 => ".L2",
        };
        format!(
            "asm volatile(\"prefetch{} [%0];\" :: \"l\"({}));",
            level, self.address
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_memory_alloc() {
        let alloc = SharedMemoryAlloc {
            name: "tile_A".to_string(),
            element_type: CudaType::Float,
            num_elements: 256,
            offset: 0,
            padding: 0,
            is_dynamic: false,
        };

        assert_eq!(alloc.total_bytes(), 1024);
        let decl = alloc.to_cuda_decl();
        assert!(decl.contains("__shared__"));
        assert!(decl.contains("tile_A[256]"));
    }

    #[test]
    fn test_memory_plan() {
        let mut plan = MemoryPlan::new();

        plan.add_shared_alloc(SharedMemoryAlloc {
            name: "smem".to_string(),
            element_type: CudaType::Float,
            num_elements: 512,
            offset: 0,
            padding: 0,
            is_dynamic: false,
        });

        assert_eq!(plan.total_static_shared, 2048);
        assert!(plan.fits_shared_memory(GpuArch::Sm80));
    }

    #[test]
    fn test_occupancy_estimate() {
        let mut plan = MemoryPlan::new();
        plan.registers_per_thread = 32;

        let occupancy = plan.estimate_occupancy(GpuArch::Sm80, 256);
        assert!(occupancy > 0.0 && occupancy <= 1.0);
    }

    #[test]
    fn test_coalescing_analysis() {
        let planner = MemoryPlanner::new(GpuArch::Sm80);

        // Sequential access
        let analysis = planner.analyze_coalescing(4, 4);
        assert!(analysis.is_coalesced);

        // Strided access
        let analysis = planner.analyze_coalescing(128, 4);
        assert!(!analysis.is_coalesced);
    }

    #[test]
    fn test_bank_conflict_analysis() {
        let planner = MemoryPlanner::new(GpuArch::Sm80);

        // Conflict-free access
        let analysis = planner.analyze_bank_conflicts(1, 4);
        assert!(analysis.is_conflict_free);

        // Column access - all threads hit same bank
        let analysis = planner.analyze_bank_conflicts(32, 4);
        assert!(!analysis.is_conflict_free);
    }

    #[test]
    fn test_matmul_tile_planning() {
        let planner = MemoryPlanner::new(GpuArch::Sm80);
        let plan = planner.plan_matmul_tiles(64, 64, 16, CudaType::Float);

        assert_eq!(plan.shared_allocations.len(), 2);
        assert!(plan.fits_shared_memory(GpuArch::Sm80));
    }

    #[test]
    fn test_tile_size_computation() {
        let planner = MemoryPlanner::new(GpuArch::Sm80);
        let max_shared = 48 * 1024; // 48 KB

        let (m, n, k) = planner.compute_tile_sizes(1024, 1024, 1024, 4, max_shared);

        // Check tiles fit
        let required = (m * k + k * n) * 4;
        assert!(required <= max_shared);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(32, 32), 32);
        assert_eq!(gcd(17, 13), 1);
    }
}
