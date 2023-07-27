#!/bin/bash

echo "Applying cargo fmt"
cargo +nightly fmt --all

echo "Checking cargo clippy"
cargo clippy --all-targets --all-features -- -D warnings

