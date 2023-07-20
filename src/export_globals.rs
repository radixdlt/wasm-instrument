use crate::parser::{
	translator::{DefaultTranslator, Translator},
	ModuleInfo,
};
use alloc::{format, vec, vec::Vec};
use wasm_encoder::{ExportKind, ExportSection, SectionId};

/// Export all declared mutable globals as `prefix_index`.
///
/// This will export all internal mutable globals under the name of
/// concat(`prefix`, `"_"`, `i`) where i is the index inside the range of
/// [0..total number of internal mutable globals].
pub fn export_mutable_globals(module_info: &mut ModuleInfo, prefix: &str) {
	let mutable_globals_to_export = module_info
		.global_section()
		.unwrap()
		.unwrap_or(vec![])
		.iter()
		.enumerate()
		.filter_map(|(index, global)| if global.ty.mutable { Some(index as u32) } else { None })
		.collect::<Vec<u32>>();

	let mut export_sec_builder = ExportSection::new();

	// Recreate current export section
	for export in module_info.export_section().unwrap().unwrap_or(vec![]) {
		let export_kind = DefaultTranslator.translate_export_kind(export.kind).unwrap();
		export_sec_builder.export(export.name, export_kind, export.index);
	}

	// Add mutable globals to the export section
	for (symbol_index, export) in mutable_globals_to_export.into_iter().enumerate() {
		let name = format!("{}_{}", prefix, symbol_index);
		export_sec_builder.export(
			&name,
			ExportKind::Global,
			module_info.imported_globals_count + export,
		);
	}

	module_info
		.replace_section(SectionId::Export.into(), &export_sec_builder)
		.unwrap();
}

#[cfg(test)]
mod tests {
	use super::export_mutable_globals;
	use crate::parser::ModuleInfo;

	fn parse_wat(source: &str) -> ModuleInfo {
		let module_bytes = wat::parse_str(source).unwrap();
		ModuleInfo::new(&module_bytes).expect("failed to parse module")
	}

	macro_rules! test_export_global {
		(name = $name:ident; input = $input:expr; expected = $expected:expr) => {
			#[test]
			fn $name() {
				let mut input_module = parse_wat($input);
				let expected_module = parse_wat($expected);

				export_mutable_globals(&mut input_module, "exported_internal_global");

				let actual_bytes = input_module.bytes();
				let expected_bytes = expected_module.bytes();

				let actual_wat = wasmprinter::print_bytes(actual_bytes).unwrap();
				let expected_wat = wasmprinter::print_bytes(expected_bytes).unwrap();

				if actual_wat != expected_wat {
					#[cfg(features = "std")]
					for diff in diff::lines(&expected_wat, &actual_wat) {
						match diff {
							diff::Result::Left(l) => println!("-{}", l),
							diff::Result::Both(l, _) => println!(" {}", l),
							diff::Result::Right(r) => println!("+{}", r),
						}
					}
					panic!()
				}
			}
		};
	}

	test_export_global! {
		name = simple;
		input = r#"
		(module
			(global (;0;) (mut i32) (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0)))
		"#;
		expected = r#"
		(module
			(global (;0;) (mut i32) (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0))
			(export "exported_internal_global_0" (global 0))
			(export "exported_internal_global_1" (global 1)))
		"#
	}

	test_export_global! {
		name = with_import;
		input = r#"
		(module
			(import "env" "global" (global $global i64))
			(global (;0;) (mut i32) (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0)))
		"#;
		expected = r#"
		(module
			(import "env" "global" (global $global i64))
			(global (;0;) (mut i32) (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0))
			(export "exported_internal_global_0" (global 1))
			(export "exported_internal_global_1" (global 2)))
		"#
	}

	test_export_global! {
		name = with_import_and_some_are_immutable;
		input = r#"
		(module
			(import "env" "global" (global $global i64))
			(global (;0;) i32 (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0)))
		"#;
		expected = r#"
		(module
			(import "env" "global" (global $global i64))
			(global (;0;) i32 (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0))
			(export "exported_internal_global_0" (global 2)))
		"#
	}
}
