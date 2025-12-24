# Contributing Guide for MedLang

Thank you for considering contributing to MedLang! This document provides guidelines for contributions.

## 🚀 How to Contribute

### Reporting Bugs

If you found a bug:

1. Check if the bug hasn't already been reported in [Issues](https://github.com/your-username/medlang/issues)
2. If not reported, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs. observed behavior
   - Compiler version (`mlc --version`)
   - Operating system and version

### Suggesting Enhancements

To suggest new features:

1. Open an issue with the `enhancement` label
2. Clearly describe:
   - The problem the feature would solve
   - How you imagine it would work
   - Usage examples
   - Impact on existing API (if any)

### Submitting Pull Requests

1. **Fork the repository**
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes** following the guidelines below
4. **Run tests**:
   ```bash
   cd compiler
   cargo test
   cargo clippy
   cargo fmt -- --check
   ```
5. **Commit your changes**:
   ```bash
   git commit -m "feat: add my feature"
   ```
6. **Push to your branch**:
   ```bash
   git push origin feature/my-feature
   ```
7. **Open a Pull Request** on GitHub

## 📝 Code Guidelines

### Rust

- Follow standard Rust style (`cargo fmt`)
- Run `cargo clippy` and fix warnings
- Add documentation for public functions
- Use descriptive names for variables and functions

### Testing

- Add tests for new features
- Keep test coverage high
- Tests should be deterministic and independent
- Use `cargo test -- --nocapture` for debugging

### Commits

Use descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or fixing tests
- `refactor:` Code refactoring
- `style:` Formatting, semicolons, etc.
- `chore:` Maintenance tasks

Example:
```
feat: add support for 3-compartment models
```

## 🏗️ Project Structure

### Compiler (`compiler/`)

- `src/` — Source code
  - `ast/` — AST definitions
  - `lexer.rs` — Tokenization
  - `parser.rs` — Parsing
  - `typeck.rs` — Type checking
  - `lower.rs` — Lowering AST → IR
  - `codegen/` — Code generation
  - `ir.rs` — Intermediate representation
- `tests/` — Tests

### Documentation (`docs/`)

- `spec/` — Formal specifications
- `guides/` — User guides
- `examples/` — Code examples
- `dev/` — Development documentation

## 🧪 Running Tests

```bash
cd compiler

# All tests
cargo test

# Specific tests
cargo test --test golden_tests
cargo test --test end_to_end

# With output
cargo test -- --nocapture

# Unit tests only
cargo test --lib

# Integration tests only
cargo test --test '*'
```

## 📚 Documentation

### Adding Documentation

- **Specifications**: Add to `docs/spec/`
- **User guides**: Add to `docs/guides/`
- **Examples**: Add to `docs/examples/`
- **Development documentation**: Add to `docs/dev/`

### Generating Documentation

```bash
cd compiler
cargo doc --open
```

## 🔍 Pull Request Checklist

Before submitting, make sure:

- [ ] Code compiles without warnings (`cargo build`)
- [ ] All tests pass (`cargo test`)
- [ ] Clippy reports no issues (`cargo clippy`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] Documentation has been updated (if needed)
- [ ] Commits follow Conventional Commits format
- [ ] Branch is up to date with `main`

## 💬 Communication

- Use issues for technical discussions
- Be respectful and constructive
- Help other contributors when possible

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT OR Apache-2.0).

---

Thank you for contributing to MedLang! 🎉
