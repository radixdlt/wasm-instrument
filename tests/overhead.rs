extern crate radix_wasm_instrument as wasm_instrument;

use std::{
	fs::{read, read_dir, ReadDir},
	path::PathBuf,
};
use wasm_instrument::{
	gas_metering::{self, host_function, mutable_global, ConstantCostRules},
	inject_stack_limiter,
	utils::module_info::ModuleInfo,
};

fn fixture_dir() -> PathBuf {
	let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	path.push("benches");
	path.push("fixtures");
	path
}

use gas_metering::Backend;
fn gas_metered_mod_len<B: Backend>(input_wasm: &[u8], backend: B) -> (Vec<u8>, usize) {
	let mut module = ModuleInfo::new(input_wasm).expect("Failed to parse WASM input");
	let wasm_bytes =
		gas_metering::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();
	let len = wasm_bytes.len();
	(wasm_bytes, len)
}

fn stack_limited_mod_len(input_wasm: &[u8]) -> (Vec<u8>, usize) {
	let mut module = ModuleInfo::new(input_wasm).expect("Failed to parse WASM input");
	let wasm_bytes = inject_stack_limiter(&mut module, 128).unwrap();
	let len = wasm_bytes.len();
	(wasm_bytes, len)
}

struct InstrumentedWasmResults {
	filename: String,
	original_module_len: usize,
	stack_limited_len: usize,
	gas_metered_host_fn_len: usize,
	gas_metered_mut_glob_len: usize,
	gas_metered_host_fn_then_stack_limited_len: usize,
	gas_metered_mut_glob_then_stack_limited_len: usize,
}

fn size_overheads_all(files: ReadDir) -> Vec<InstrumentedWasmResults> {
	files
		.map(|entry| {
			let entry = entry.unwrap();
			let filename = entry.file_name().into_string().unwrap();

			let (original_module_len, orig_wasm) = {
				let bytes = match entry.path().extension().unwrap().to_str() {
					Some("wasm") => read(entry.path()).unwrap(),
					Some("wat") =>
						wat::parse_bytes(&read(entry.path()).unwrap()).unwrap().into_owned(),
					_ => panic!("expected fixture_dir containing .wasm or .wat files only"),
				};

				let len = bytes.len();
				(len, bytes)
			};

			let (gm_host_fn_wasm, gas_metered_host_fn_len) =
				gas_metered_mod_len(&orig_wasm, host_function::Injector::new("env", "gas"));

			let (gm_mut_global_wasm, gas_metered_mut_glob_len) =
				gas_metered_mod_len(&orig_wasm, mutable_global::Injector::new("env", "gas_left"));

			let stack_limited_len = stack_limited_mod_len(&orig_wasm).1;

			let (_gm_hf_sl_mod, gas_metered_host_fn_then_stack_limited_len) =
				stack_limited_mod_len(&gm_host_fn_wasm);

			let (_gm_mg_sl_module, gas_metered_mut_glob_then_stack_limited_len) =
				stack_limited_mod_len(&gm_mut_global_wasm);

			InstrumentedWasmResults {
				filename,
				original_module_len,
				stack_limited_len,
				gas_metered_host_fn_len,
				gas_metered_mut_glob_len,
				gas_metered_host_fn_then_stack_limited_len,
				gas_metered_mut_glob_then_stack_limited_len,
			}
		})
		.collect()
}

fn calc_size_overheads() -> Vec<InstrumentedWasmResults> {
	let mut wasm_path = fixture_dir();
	wasm_path.push("wasm");

	let mut wat_path = fixture_dir();
	wat_path.push("wat");

	let mut results = size_overheads_all(read_dir(wasm_path).unwrap());
	let results_wat = size_overheads_all(read_dir(wat_path).unwrap());

	results.extend(results_wat);

	results
}

/// Print the overhead of applying gas metering, stack
/// height limiting or both.
///
/// Use `cargo test print_size_overhead -- --nocapture`.
#[test]
fn print_size_overhead() {
	let mut results = calc_size_overheads();
	results.sort_unstable_by(|a, b| {
		b.gas_metered_mut_glob_then_stack_limited_len
			.cmp(&a.gas_metered_mut_glob_then_stack_limited_len)
	});

	for r in results {
		let filename = r.filename;
		let original_size = r.original_module_len / 1024;
		let stack_limit = r.stack_limited_len * 100 / r.original_module_len;
		let host_fn = r.gas_metered_host_fn_len * 100 / r.original_module_len;
		let mut_glob = r.gas_metered_mut_glob_len * 100 / r.original_module_len;
		let host_fn_sl = r.gas_metered_host_fn_then_stack_limited_len * 100 / r.original_module_len;
		let mut_glob_sl =
			r.gas_metered_mut_glob_then_stack_limited_len * 100 / r.original_module_len;

		println!(
			"{filename:30}: orig = {original_size:4} kb, stack_limiter = {stack_limit} %, \
			  gas_metered_host_fn =    {host_fn} %, both = {host_fn_sl} %,\n \
			 {:69} gas_metered_mut_global = {mut_glob} %, both = {mut_glob_sl} %",
			""
		);
	}
}

/// Compare module size overhead of applying gas metering with two methods.
///
/// Use `cargo test print_gas_metered_sizes -- --nocapture`.
#[test]
fn print_gas_metered_sizes() {
	let overheads = calc_size_overheads();
	let mut results = overheads
		.iter()
		.map(|r| {
			let diff = (r.gas_metered_mut_glob_len * 100 / r.gas_metered_host_fn_len) as i32 - 100;
			(diff, r)
		})
		.collect::<Vec<(i32, &InstrumentedWasmResults)>>();
	results.sort_unstable_by(|a, b| b.0.cmp(&a.0));

	println!(
		"| {:28} | {:^16} | gas metered/host fn | gas metered/mut global | size diff |",
		"fixture", "original size",
	);
	println!("|{:-^30}|{:-^18}|{:-^21}|{:-^24}|{:-^11}|", "", "", "", "", "",);
	for r in results {
		let filename = &r.1.filename;
		let original_size = &r.1.original_module_len / 1024;
		let host_fn = &r.1.gas_metered_host_fn_len / 1024;
		let mut_glob = &r.1.gas_metered_mut_glob_len / 1024;
		let host_fn_percent = &r.1.gas_metered_host_fn_len * 100 / r.1.original_module_len;
		let mut_glob_percent = &r.1.gas_metered_mut_glob_len * 100 / r.1.original_module_len;
		let host_fn = format!("{host_fn} kb ({host_fn_percent:}%)");
		let mut_glob = format!("{mut_glob} kb ({mut_glob_percent:}%)");
		let diff = &r.0;
		println!(
			"| {filename:28} | {original_size:13} kb | {host_fn:>19} | {mut_glob:>22} | {diff:+8}% |"
		);
	}
}
