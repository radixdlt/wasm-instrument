//! Wasm binary graph format

use parity_wasm::elements;
use super::ref_list::{RefList, EntryRef};
use std::vec::Vec;
use std::borrow::ToOwned;
use std::string::String;

enum ImportedOrDeclared<T=()> {
	Imported(String, String),
	Declared(T),
}

impl<T> From<&elements::ImportEntry> for ImportedOrDeclared<T> {
	fn from(v: &elements::ImportEntry) -> Self {
		ImportedOrDeclared::Imported(v.module().to_owned(), v.field().to_owned())
	}
}

type FuncOrigin = ImportedOrDeclared<Vec<Instruction>>;
type GlobalOrigin = ImportedOrDeclared<Vec<Instruction>>;
type MemoryOrigin = ImportedOrDeclared;
type TableOrigin = ImportedOrDeclared;

struct Func {
	type_ref: EntryRef<elements::Type>,
	origin: FuncOrigin,
}

struct Global {
	content: elements::ValueType,
	is_mut: bool,
	origin: GlobalOrigin,
}

enum Instruction {
	Plain(elements::Instruction),
	Call(EntryRef<Func>),
}

struct Memory {
	limits: elements::ResizableLimits,
	origin: MemoryOrigin,
}

struct Table {
	origin: TableOrigin,
	limits: elements::ResizableLimits,
}

struct DataSegment {
	offset_expr: Vec<Instruction>,
	data: Vec<u8>,
}

struct ElementSegment {
	offset_expr: Vec<Instruction>,
	data: Vec<u32>,
}

enum Export {
	Func(EntryRef<Func>),
	Global(EntryRef<Global>),
	Table(EntryRef<Table>),
	Memory(EntryRef<Memory>),
}

#[derive(Default)]
struct Module {
	types: RefList<elements::Type>,
	funcs: RefList<Func>,
	memory: RefList<Memory>,
	tables: RefList<Table>,
	globals: RefList<Global>,
	start: Option<EntryRef<Func>>,
	exports: Vec<Export>,
	elements: Vec<ElementSegment>,
	data: Vec<DataSegment>,
}

impl Module {

	fn from_elements(module: &elements::Module) -> Self {

		let mut res = Module::default();

		for section in module.sections() {
			match section {
				elements::Section::Type(type_section) => {
					res.types = RefList::from_slice(type_section.types());
				},
				elements::Section::Import(import_section) => {
					for entry in import_section.entries() {
						match *entry.external() {
							elements::External::Function(f) => {
								res.funcs.push(Func {
									type_ref: res.types.get(f as usize).expect("validated; qed").clone(),
									origin: entry.into(),
								});
							},
							elements::External::Memory(m) => {
								res.memory.push(Memory {
									limits: m.limits().clone(),
									origin: entry.into(),
								});
							},
							elements::External::Global(g) => {
								res.globals.push(Global {
									content: g.content_type(),
									is_mut: g.is_mutable(),
									origin: entry.into(),
								});
							},
							elements::External::Table(t) => {
								res.tables.push(Table {
									limits: t.limits().clone(),
									origin: entry.into(),
								});
							},
						};
					}
				},
				elements::Section::Function(function_section) => {
					for f in function_section.entries() {
						res.funcs.push(Func {
							type_ref: res.types.get(f.type_ref() as usize).expect("validated; qed").clone(),
							// code will be populated later
							origin: ImportedOrDeclared::Declared(Vec::new()),
						});
					};
				},
				elements::Section::Table(table_section) => {
					for t in table_section.entries() {
						res.tables.push(Table {
							limits: t.limits().clone(),
							origin: ImportedOrDeclared::Declared(()),
						});
					}
				},
				elements::Section::Memory(table_section) => {
					for t in table_section.entries() {
						res.memory.push(Memory {
							limits: t.limits().clone(),
							origin: ImportedOrDeclared::Declared(()),
						});
					}
				},
				elements::Section::Global(global_section) => {
					for g in global_section.entries() {
						res.globals.push(Global {
							content: g.global_type().content_type(),
							is_mut: g.global_type().is_mutable(),
							// TODO: init expr
							origin: ImportedOrDeclared::Declared(Vec::new()),
						});
					}
				},
				elements::Section::Export(export_section) => {
					for e in export_section.entries() {
						match e.internal() {
							&elements::Internal::Function(func_idx) => {
								res.exports.push(Export::Func(res.funcs.clone_ref(func_idx as usize)));
							},
							&elements::Internal::Global(global_idx) => {
								res.exports.push(Export::Global(res.globals.clone_ref(global_idx as usize)));
							},
							&elements::Internal::Memory(mem_idx) => {
								res.exports.push(Export::Memory(res.memory.clone_ref(mem_idx as usize)));
							},
							&elements::Internal::Table(table_idx) => {
								res.exports.push(Export::Table(res.tables.clone_ref(table_idx as usize)));
							},
						}
					}
				},
				_ => continue,
			}
		}

		res
	}

}

fn parse(wasm: &[u8]) -> Module {
	Module::from_elements(&::parity_wasm::deserialize_buffer(wasm).expect("failed to parse wasm"))
}

#[cfg(test)]
mod tests {

	extern crate wabt;
	use parity_wasm;

	#[test]
	fn smoky() {
		let wasm = wabt::wat2wasm(r#"
			(module
				(type (func))
				(func (type 0))
				(memory 0 1)
				(export "simple" (func 0))
			)
		"#).expect("Failed to read fixture");

		let f = super::parse(&wasm[..]);

		assert_eq!(f.types.len(), 1);
		assert_eq!(f.funcs.len(), 1);
		assert_eq!(f.tables.len(), 0);
		assert_eq!(f.memory.len(), 1);
		assert_eq!(f.exports.len(), 1);

		assert_eq!(f.types.get_ref(0).link_count(), 1);
		assert_eq!(f.funcs.get_ref(0).link_count(), 1);
	}
}