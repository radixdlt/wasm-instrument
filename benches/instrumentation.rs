use criterion::{
	criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, Criterion,
	Throughput,
};
use std::{
	fs::{read, read_dir},
	path::PathBuf,
};
use wasm_instrument::{
	gas_metering::{self, host_function, ConstantCostRules},
	inject_stack_limiter,
	utils::module_info::ModuleInfo,
};

fn fixture_dir() -> PathBuf {
	let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	path.push("benches");
	path.push("fixtures");
	path.push("wasm");
	path
}

fn for_fixtures<F, M>(group: &mut BenchmarkGroup<M>, f: F)
where
	F: Fn(ModuleInfo),
	M: Measurement,
{
	for entry in read_dir(fixture_dir()).unwrap() {
		let entry = entry.unwrap();
		let bytes = read(entry.path()).unwrap();
		group.throughput(Throughput::Bytes(bytes.len().try_into().unwrap()));
		group.bench_with_input(entry.file_name().to_str().unwrap(), &bytes, |bench, input| {
			bench.iter(|| f(ModuleInfo::new(input).unwrap()))
		});
	}
}

fn gas_metering(c: &mut Criterion) {
	let mut group = c.benchmark_group("Gas Metering");
	for_fixtures(&mut group, |mut module| {
		gas_metering::inject(
			&mut module,
			host_function::Injector::new("env", "gas"),
			&ConstantCostRules::default(),
		)
		.unwrap();
	});
}

fn stack_height_limiter(c: &mut Criterion) {
	let mut group = c.benchmark_group("Stack Height Limiter");
	for_fixtures(&mut group, |mut module| {
		inject_stack_limiter(&mut module, 128).unwrap();
	});
}

criterion_group!(benches, gas_metering, stack_height_limiter);
criterion_main!(benches);
