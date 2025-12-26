# Migration Status

## ✅ Completed

- [x] Inventory of MedLang standalone
- [x] Unification plan document
- [x] Archive notices and documentation
- [x] Migration package created
- [x] PK models migrated (one/two compartment)
- [x] Dosing protocols migrated
- [x] Policies migrated
- [x] Migration guide created

## 📦 Migration Package

Location: `migration/sounio-structure/stdlib/medlang/`

Files ready for Sounio:
- `pk/one_compartment.sio` - One-compartment PK models
- `pk/two_compartment.sio` - Two-compartment PK models
- `pk/mod.sio` - PK module exports
- `dose/mod.sio` - Dosing protocols
- `policy/mod.sio` - Dosing policies
- `mod.sio` - Main module

## ⬜ Pending (when Sounio available)

- [ ] Create structure in Sounio repository
- [ ] Copy files from migration package
- [ ] Adapt syntax to actual Sounio syntax
- [ ] Add remaining modules (PD, PBPK, population, etc.)
- [ ] Convert examples
- [ ] Test compilation
- [ ]rate into Sounio documentation

## 📝 Notes

- Files use proposed Sounio syntax (may need adjustment)
- All models use `Knowledge<T>` for uncertainty
- Compartment/flow syntax is proposed
- Ready for integration when Sounio repository structure is available
