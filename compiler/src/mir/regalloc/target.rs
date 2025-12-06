//! Target-Specific Register Information
//!
//! Defines physical registers and register classes for different architectures.

use std::fmt;

/// A physical register on the target machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalRegister {
    /// Register number
    pub number: u8,
    /// Register class
    pub class: RegisterClass,
}

impl PhysicalRegister {
    /// Create a new physical register
    pub fn new(number: u8, class: RegisterClass) -> Self {
        Self { number, class }
    }

    /// Get register name for x86-64
    pub fn x86_64_name(&self) -> &'static str {
        match self.class {
            RegisterClass::GPR => match self.number {
                0 => "rax",
                1 => "rcx",
                2 => "rdx",
                3 => "rbx",
                4 => "rsp",
                5 => "rbp",
                6 => "rsi",
                7 => "rdi",
                8 => "r8",
                9 => "r9",
                10 => "r10",
                11 => "r11",
                12 => "r12",
                13 => "r13",
                14 => "r14",
                15 => "r15",
                _ => "?gpr",
            },
            RegisterClass::FPR => match self.number {
                0..=15 => {
                    const NAMES: [&str; 16] = [
                        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8",
                        "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
                    ];
                    NAMES[self.number as usize]
                }
                _ => "?xmm",
            },
            RegisterClass::Vector => match self.number {
                0..=15 => {
                    const NAMES: [&str; 16] = [
                        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
                        "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
                    ];
                    NAMES[self.number as usize]
                }
                _ => "?ymm",
            },
            RegisterClass::Flags => "flags",
        }
    }

    /// Get register name for AArch64
    pub fn aarch64_name(&self) -> &'static str {
        match self.class {
            RegisterClass::GPR => {
                if self.number < 31 {
                    const NAMES: [&str; 31] = [
                        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
                        "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21",
                        "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
                    ];
                    NAMES[self.number as usize]
                } else if self.number == 31 {
                    "sp"
                } else {
                    "?x"
                }
            }
            RegisterClass::FPR | RegisterClass::Vector => {
                if self.number < 32 {
                    const NAMES: [&str; 32] = [
                        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                        "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
                        "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                    ];
                    NAMES[self.number as usize]
                } else {
                    "?v"
                }
            }
            RegisterClass::Flags => "nzcv",
        }
    }
}

impl fmt::Display for PhysicalRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.x86_64_name())
    }
}

/// Register class (category of registers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterClass {
    /// General-purpose registers (integers, pointers)
    GPR,
    /// Floating-point registers
    FPR,
    /// Vector/SIMD registers
    Vector,
    /// Flags register
    Flags,
}

impl RegisterClass {
    /// Get the size of registers in this class (in bytes)
    pub fn size(&self) -> u32 {
        match self {
            RegisterClass::GPR => 8,     // 64-bit
            RegisterClass::FPR => 8,     // 64-bit double
            RegisterClass::Vector => 32, // 256-bit AVX
            RegisterClass::Flags => 8,
        }
    }
}

/// Information about target registers
#[derive(Debug, Clone)]
pub struct RegisterInfo {
    /// Register
    pub reg: PhysicalRegister,
    /// Is this a callee-saved register?
    pub callee_saved: bool,
    /// Is this reserved (stack pointer, etc.)?
    pub reserved: bool,
    /// Aliases (overlapping registers)
    pub aliases: Vec<PhysicalRegister>,
}

/// Target register configuration
#[derive(Debug, Clone)]
pub struct TargetRegisters {
    /// All available registers
    pub registers: Vec<RegisterInfo>,
    /// Number of GPRs
    pub num_gprs: usize,
    /// Number of FPRs
    pub num_fprs: usize,
    /// Number of vector registers
    pub num_vectors: usize,
    /// Stack pointer register
    pub stack_pointer: PhysicalRegister,
    /// Frame pointer register
    pub frame_pointer: PhysicalRegister,
    /// Return address register (if applicable)
    pub return_address: Option<PhysicalRegister>,
}

impl TargetRegisters {
    /// Create x86-64 register configuration
    pub fn x86_64() -> Self {
        let mut registers = Vec::new();

        // GPRs (0-15)
        let callee_saved_gprs = [3, 5, 12, 13, 14, 15]; // rbx, rbp, r12-r15
        let reserved_gprs = [4, 5]; // rsp, rbp

        for i in 0..16u8 {
            registers.push(RegisterInfo {
                reg: PhysicalRegister::new(i, RegisterClass::GPR),
                callee_saved: callee_saved_gprs.contains(&i),
                reserved: reserved_gprs.contains(&i),
                aliases: vec![],
            });
        }

        // XMM registers (0-15)
        for i in 0..16u8 {
            registers.push(RegisterInfo {
                reg: PhysicalRegister::new(i, RegisterClass::FPR),
                callee_saved: false, // All caller-saved on System V AMD64
                reserved: false,
                aliases: vec![PhysicalRegister::new(i, RegisterClass::Vector)],
            });
        }

        // YMM registers (0-15) - aliases with XMM
        for i in 0..16u8 {
            registers.push(RegisterInfo {
                reg: PhysicalRegister::new(i, RegisterClass::Vector),
                callee_saved: false,
                reserved: false,
                aliases: vec![PhysicalRegister::new(i, RegisterClass::FPR)],
            });
        }

        Self {
            registers,
            num_gprs: 16,
            num_fprs: 16,
            num_vectors: 16,
            stack_pointer: PhysicalRegister::new(4, RegisterClass::GPR),
            frame_pointer: PhysicalRegister::new(5, RegisterClass::GPR),
            return_address: None, // x86-64 uses stack for return address
        }
    }

    /// Create AArch64 register configuration
    pub fn aarch64() -> Self {
        let mut registers = Vec::new();

        // GPRs (x0-x30, sp)
        let callee_saved_gprs: Vec<u8> = (19..=28).collect(); // x19-x28
        let reserved_gprs = [29, 30, 31]; // x29=fp, x30=lr, x31=sp/zr

        for i in 0..32u8 {
            registers.push(RegisterInfo {
                reg: PhysicalRegister::new(i, RegisterClass::GPR),
                callee_saved: callee_saved_gprs.contains(&i),
                reserved: reserved_gprs.contains(&i),
                aliases: vec![],
            });
        }

        // SIMD/FP registers (v0-v31)
        let callee_saved_fprs: Vec<u8> = (8..=15).collect(); // v8-v15

        for i in 0..32u8 {
            registers.push(RegisterInfo {
                reg: PhysicalRegister::new(i, RegisterClass::FPR),
                callee_saved: callee_saved_fprs.contains(&i),
                reserved: false,
                aliases: vec![PhysicalRegister::new(i, RegisterClass::Vector)],
            });
        }

        Self {
            registers,
            num_gprs: 31, // x0-x30, not counting sp
            num_fprs: 32,
            num_vectors: 32,
            stack_pointer: PhysicalRegister::new(31, RegisterClass::GPR),
            frame_pointer: PhysicalRegister::new(29, RegisterClass::GPR),
            return_address: Some(PhysicalRegister::new(30, RegisterClass::GPR)),
        }
    }

    /// Create RISC-V 64 register configuration
    pub fn riscv64() -> Self {
        let mut registers = Vec::new();

        // GPRs (x0-x31)
        // x0 = zero, x1 = ra, x2 = sp, x8 = fp
        let callee_saved_gprs: Vec<u8> = vec![8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
        let reserved_gprs = [0, 2, 3, 4]; // zero, sp, gp, tp

        for i in 0..32u8 {
            registers.push(RegisterInfo {
                reg: PhysicalRegister::new(i, RegisterClass::GPR),
                callee_saved: callee_saved_gprs.contains(&i),
                reserved: reserved_gprs.contains(&i),
                aliases: vec![],
            });
        }

        // FP registers (f0-f31)
        let callee_saved_fprs: Vec<u8> = (8..=9).chain(18..=27).collect();

        for i in 0..32u8 {
            registers.push(RegisterInfo {
                reg: PhysicalRegister::new(i, RegisterClass::FPR),
                callee_saved: callee_saved_fprs.contains(&i),
                reserved: false,
                aliases: vec![],
            });
        }

        Self {
            registers,
            num_gprs: 32,
            num_fprs: 32,
            num_vectors: 0, // Base RV64 doesn't have vector
            stack_pointer: PhysicalRegister::new(2, RegisterClass::GPR),
            frame_pointer: PhysicalRegister::new(8, RegisterClass::GPR),
            return_address: Some(PhysicalRegister::new(1, RegisterClass::GPR)),
        }
    }

    /// Get allocatable registers for a class
    pub fn allocatable(&self, class: RegisterClass) -> Vec<PhysicalRegister> {
        self.registers
            .iter()
            .filter(|r| r.reg.class == class && !r.reserved)
            .map(|r| r.reg)
            .collect()
    }

    /// Get callee-saved registers for a class
    pub fn callee_saved(&self, class: RegisterClass) -> Vec<PhysicalRegister> {
        self.registers
            .iter()
            .filter(|r| r.reg.class == class && r.callee_saved)
            .map(|r| r.reg)
            .collect()
    }

    /// Get caller-saved registers for a class
    pub fn caller_saved(&self, class: RegisterClass) -> Vec<PhysicalRegister> {
        self.registers
            .iter()
            .filter(|r| r.reg.class == class && !r.callee_saved && !r.reserved)
            .map(|r| r.reg)
            .collect()
    }

    /// Get all registers for a class
    pub fn registers_for_class(&self, class: &RegisterClass) -> Vec<PhysicalRegister> {
        self.allocatable(*class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x86_64_registers() {
        let target = TargetRegisters::x86_64();

        assert_eq!(target.num_gprs, 16);
        assert_eq!(target.num_fprs, 16);

        let allocatable_gprs = target.allocatable(RegisterClass::GPR);
        // Should exclude rsp and rbp
        assert_eq!(allocatable_gprs.len(), 14);
    }

    #[test]
    fn test_aarch64_registers() {
        let target = TargetRegisters::aarch64();

        assert_eq!(target.num_gprs, 31);
        assert_eq!(target.num_fprs, 32);

        let callee_saved = target.callee_saved(RegisterClass::GPR);
        assert_eq!(callee_saved.len(), 10); // x19-x28
    }

    #[test]
    fn test_register_names() {
        let rax = PhysicalRegister::new(0, RegisterClass::GPR);
        assert_eq!(rax.x86_64_name(), "rax");

        let x0 = PhysicalRegister::new(0, RegisterClass::GPR);
        assert_eq!(x0.aarch64_name(), "x0");
    }
}
