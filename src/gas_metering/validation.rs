//! This module is used to validate the correctness of the gas metering algorithm.
//!
//! Since the gas metering algorithm is complex, this checks correctness by fuzzing. The testing
//! strategy is to generate random, valid Wasm modules using Binaryen's translate-to-fuzz
//! functionality, then ensure for all functions defined, in all execution paths though the
//! function body that do not trap that the amount of gas charged by the proposed metering
//! instructions is correct. This is done by constructing a control flow graph and exhaustively
//! searching through all paths, which may take exponential time in the size of the function body in
//! the worst case.

use super::{MeteredBlock, Rules};
use anyhow::{anyhow, Result};
use std::collections::BTreeMap as Map;
use wasmparser::Operator;

/// An ID for a node in a ControlFlowGraph.
type NodeId = usize;

/// A node in a control flow graph is commonly known as a basic block. This is a sequence of
/// operations that are always executed sequentially.
#[derive(Debug, Default)]
struct ControlFlowNode {
	/// The index of the first instruction in the basic block. This is only used for debugging.
	first_instr_pos: Option<usize>,

	/// The actual gas cost of executing all instructions in the basic block.
	actual_cost: u64,

	/// The amount of gas charged by the injected metering instructions within this basic block.
	charged_cost: u64,

	/// Whether there are any other nodes in the graph that loop back to this one. Every cycle in
	/// the control flow graph contains at least one node with this flag set.
	is_loop_target: bool,

	/// Edges in the "forward" direction of the graph. The graph of nodes and their forward edges
	/// forms a directed acyclic graph (DAG).
	forward_edges: Vec<NodeId>,

	/// Edges in the "backwards" direction. These edges form cycles in the graph.
	loopback_edges: Vec<NodeId>,
}

/// A control flow graph where nodes are basic blocks and edges represent possible transitions
/// between them in execution flow. The graph has two types of edges, forward and loop-back edges.
/// The subgraph with only the forward edges forms a directed acyclic graph (DAG); including the
/// loop-back edges introduces cycles.
#[derive(Debug)]
pub struct ControlFlowGraph {
	nodes: Vec<ControlFlowNode>,
}

impl ControlFlowGraph {
	fn new() -> Self {
		ControlFlowGraph { nodes: Vec::new() }
	}

	fn get_node(&self, node_id: NodeId) -> &ControlFlowNode {
		self.nodes.get(node_id).unwrap()
	}

	fn get_node_mut(&mut self, node_id: NodeId) -> &mut ControlFlowNode {
		self.nodes.get_mut(node_id).unwrap()
	}

	fn add_node(&mut self) -> NodeId {
		self.nodes.push(ControlFlowNode::default());
		self.nodes.len() - 1
	}

	fn increment_actual_cost(&mut self, node_id: NodeId, cost: u64) {
		self.get_node_mut(node_id).actual_cost += cost;
	}

	fn increment_charged_cost(&mut self, node_id: NodeId, cost: u64) {
		self.get_node_mut(node_id).charged_cost += cost;
	}

	fn set_first_instr_pos(&mut self, node_id: NodeId, first_instr_pos: usize) {
		self.get_node_mut(node_id).first_instr_pos = Some(first_instr_pos)
	}

	fn new_edge(&mut self, from_id: NodeId, target_frame: &ControlFrame) {
		if target_frame.is_loop {
			self.new_loopback_edge(from_id, target_frame.entry_node);
		} else {
			self.new_forward_edge(from_id, target_frame.exit_node);
		}
	}

	fn new_forward_edge(&mut self, from_id: NodeId, to_id: NodeId) {
		self.get_node_mut(from_id).forward_edges.push(to_id)
	}

	fn new_loopback_edge(&mut self, from_id: NodeId, to_id: NodeId) {
		self.get_node_mut(from_id).loopback_edges.push(to_id);
		self.get_node_mut(to_id).is_loop_target = true;
	}
}

/// A control frame is opened upon entry into a function and by the `block`, `if`, and `loop`
/// instructions and is closed by `end` instructions.
struct ControlFrame {
	is_loop: bool,
	entry_node: NodeId,
	exit_node: NodeId,
	active_node: NodeId,
}

impl ControlFrame {
	fn new(entry_node_id: NodeId, exit_node_id: NodeId, is_loop: bool) -> Self {
		ControlFrame {
			is_loop,
			entry_node: entry_node_id,
			exit_node: exit_node_id,
			active_node: entry_node_id,
		}
	}
}

/// Construct a control flow graph from a function body and the metered blocks computed for it.
///
/// This assumes that the function body has been validated already, otherwise this may panic.
fn build_control_flow_graph(
	body: &wasmparser::FunctionBody,
	rules: &impl Rules,
	blocks: &[MeteredBlock],
	locals_count: u32,
) -> Result<ControlFlowGraph> {
	use wasmparser::Operator::*;

	let mut graph = ControlFlowGraph::new();

	let entry_node_id = graph.add_node();
	let terminal_node_id = graph.add_node();

	graph.set_first_instr_pos(entry_node_id, 0);

	let mut stack = vec![ControlFrame::new(entry_node_id, terminal_node_id, false)];
	let mut metered_blocks_iter = blocks.iter().peekable();

	let locals_init_cost = rules.call_per_local_cost().checked_mul(locals_count).ok_or(()).unwrap();

	let operators = body
		.get_operators_reader()?
		.into_iter()
		.collect::<wasmparser::Result<Vec<Operator>>>()?;
	for (cursor, instruction) in operators.iter().enumerate() {
		let active_node_id = stack
			.last()
			.expect("module is valid by pre-condition; control stack must not be empty; qed")
			.active_node;

		// Increment the charged cost if there are metering instructions to be inserted here.
		let apply_block =
			metered_blocks_iter.peek().map_or(false, |block| block.start_pos == cursor);
		if apply_block {
			let next_metered_block =
				metered_blocks_iter.next().expect("peek returned an item; qed");
			graph.increment_charged_cost(active_node_id, next_metered_block.cost);
		}

		// Add locals initialization cost to the function block.
		if cursor == 0 {
			graph.increment_actual_cost(active_node_id, locals_init_cost.into());
		}

		let instruction_cost = rules
			.instruction_cost(instruction)
			.ok_or_else(|| anyhow!("gas rule for instruction {:?} not found", &instruction))?
			as u64;
		match instruction {
			Block { .. } => {
				graph.increment_actual_cost(active_node_id, instruction_cost);

				let exit_node_id = graph.add_node();
				stack.push(ControlFrame::new(active_node_id, exit_node_id, false));
			},
			If { .. } => {
				graph.increment_actual_cost(active_node_id, instruction_cost);

				let then_node_id = graph.add_node();
				let exit_node_id = graph.add_node();

				stack.push(ControlFrame::new(then_node_id, exit_node_id, false));
				graph.new_forward_edge(active_node_id, then_node_id);
				graph.set_first_instr_pos(then_node_id, cursor + 1);
			},
			Loop { .. } => {
				graph.increment_actual_cost(active_node_id, instruction_cost);

				let loop_node_id = graph.add_node();
				let exit_node_id = graph.add_node();

				stack.push(ControlFrame::new(loop_node_id, exit_node_id, true));
				graph.new_forward_edge(active_node_id, loop_node_id);
				graph.set_first_instr_pos(loop_node_id, cursor + 1);
			},
			Else => {
				let active_frame_idx = stack.len() - 1;
				let prev_frame_idx = stack.len() - 2;

				let else_node_id = graph.add_node();
				stack[active_frame_idx].active_node = else_node_id;

				let prev_node_id = stack[prev_frame_idx].active_node;
				graph.new_forward_edge(prev_node_id, else_node_id);
				graph.set_first_instr_pos(else_node_id, cursor + 1);
			},
			End => {
				let closing_frame = stack.pop()
					.expect("module is valid by pre-condition; ends correspond to control stack frames; qed");

				graph.new_forward_edge(active_node_id, closing_frame.exit_node);
				graph.set_first_instr_pos(closing_frame.exit_node, cursor + 1);

				if let Some(active_frame) = stack.last_mut() {
					active_frame.active_node = closing_frame.exit_node;
				}
			},
			Br { relative_depth: label } => {
				graph.increment_actual_cost(active_node_id, instruction_cost);

				let active_frame_idx = stack.len() - 1;
				let target_frame_idx = active_frame_idx - (*label as usize);
				graph.new_edge(active_node_id, &stack[target_frame_idx]);

				// Next instruction is unreachable, but carry on anyway.
				let new_node_id = graph.add_node();
				stack[active_frame_idx].active_node = new_node_id;
				graph.set_first_instr_pos(new_node_id, cursor + 1);
			},
			BrIf { relative_depth: label } => {
				graph.increment_actual_cost(active_node_id, instruction_cost);

				let active_frame_idx = stack.len() - 1;
				let target_frame_idx = active_frame_idx - (*label as usize);
				graph.new_edge(active_node_id, &stack[target_frame_idx]);

				let new_node_id = graph.add_node();
				stack[active_frame_idx].active_node = new_node_id;
				graph.new_forward_edge(active_node_id, new_node_id);
				graph.set_first_instr_pos(new_node_id, cursor + 1);
			},
			BrTable { targets: br_table_data } => {
				graph.increment_actual_cost(active_node_id, instruction_cost);

				let active_frame_idx = stack.len() - 1;

				let r = br_table_data.targets().collect::<wasmparser::Result<Vec<u32>>>()?;
				for label in [br_table_data.default()].iter().chain(r.iter()) {
					let target_frame_idx = active_frame_idx - (*label as usize);
					graph.new_edge(active_node_id, &stack[target_frame_idx]);
				}

				let new_node_id = graph.add_node();
				stack[active_frame_idx].active_node = new_node_id;
				graph.set_first_instr_pos(new_node_id, cursor + 1);
			},
			Return => {
				graph.increment_actual_cost(active_node_id, instruction_cost);

				graph.new_forward_edge(active_node_id, terminal_node_id);

				let active_frame_idx = stack.len() - 1;
				let new_node_id = graph.add_node();
				stack[active_frame_idx].active_node = new_node_id;
				graph.set_first_instr_pos(new_node_id, cursor + 1);
			},
			_ => graph.increment_actual_cost(active_node_id, instruction_cost),
		}
	}

	assert!(stack.is_empty());

	Ok(graph)
}

/// Exhaustively search through all paths in the control flow graph, starting from the first node
/// and ensure that 1) all paths with only forward edges ending with the terminal node have an
/// equal total actual gas cost and total charged gas cost, and 2) all cycles beginning with a loop
/// entry point and ending with a node with a loop-back edge to the entry point have equal actual
/// and charged gas costs. If this returns true, then the metered blocks used to construct the
/// control flow graph are correct with respect to the function body.
///
/// In the worst case, this runs in time exponential in the size of the graph.
fn validate_graph_gas_costs(graph: &ControlFlowGraph) -> bool {
	fn visit(
		graph: &ControlFlowGraph,
		node_id: NodeId,
		mut total_actual: u64,
		mut total_charged: u64,
		loop_costs: &mut Map<NodeId, (u64, u64)>,
	) -> bool {
		let node = graph.get_node(node_id);

		total_actual += node.actual_cost;
		total_charged += node.charged_cost;

		if node.is_loop_target {
			loop_costs.insert(node_id, (node.actual_cost, node.charged_cost));
		}

		if node.forward_edges.is_empty() && total_actual != total_charged {
			return false
		}

		for loop_node_id in node.loopback_edges.iter() {
			let (loop_actual, loop_charged) = loop_costs
				.get_mut(loop_node_id)
				.expect("cannot arrive at loopback edge without visiting loop entry node");
			if loop_actual != loop_charged {
				return false
			}
		}

		for next_node_id in node.forward_edges.iter() {
			if !visit(graph, *next_node_id, total_actual, total_charged, loop_costs) {
				return false
			}
		}

		if node.is_loop_target {
			loop_costs.remove(&node_id);
		}

		true
	}

	// Recursively explore all paths through the execution graph starting from the entry node.
	visit(graph, 0, 0, 0, &mut Map::new())
}

/// Validate that the metered blocks are correct with respect to the function body by exhaustively
/// searching all paths through the control flow graph.
///
/// This assumes that the function body has been validated already, otherwise this may panic.
fn validate_metering_injections(
	body: &wasmparser::FunctionBody,
	rules: &impl Rules,
	blocks: &[MeteredBlock],
	locals_count: u32,
) -> Result<bool> {
	let graph = build_control_flow_graph(body, rules, blocks, locals_count)?;
	Ok(validate_graph_gas_costs(&graph))
}

mod tests {
	use super::{
		super::{copy_locals, determine_metered_blocks, ConstantCostRules},
		validate_metering_injections,
	};
	use binaryen::tools::translate_to_fuzz_mvp;
	use rand::{thread_rng, RngCore};
	use wasmparser::{CodeSectionReader, FunctionBody, Payload::CodeSectionStart};

	#[test]
	fn test_build_control_flow_graph() {
		for _ in 0..20 {
			let mut rand_input = [0u8; 2048];
			thread_rng().fill_bytes(&mut rand_input);

			let module_bytes = translate_to_fuzz_mvp(&rand_input).write();
			let payload = wasmparser::Parser::new(0)
				.parse_all(&module_bytes)
				.map(|v| v.unwrap())
				.find(|payload| matches!(payload, CodeSectionStart { .. }))
				.unwrap();

			if let CodeSectionStart { range, .. } = payload {
				let reader = CodeSectionReader::new(&module_bytes[range], 0).unwrap();
				let bodies =
					reader.into_iter().collect::<wasmparser::Result<Vec<FunctionBody>>>().unwrap();
				for func_body in bodies {
					let current_locals = copy_locals(&func_body).unwrap();
					let locals_count = current_locals.iter().map(|(count, _)| count).sum();
					let rules = ConstantCostRules::default();
					let metered_blocks =
						determine_metered_blocks(&func_body, &rules, locals_count).unwrap();
					let success = validate_metering_injections(
						&func_body,
						&rules,
						&metered_blocks,
						locals_count,
					)
					.unwrap();
					assert!(success);
				}
			}
		}
	}
}
