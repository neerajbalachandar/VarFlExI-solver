#!/usr/bin/env julia

# Compatibility entry point for the standalone fluid-only resolution study.
include(joinpath(@__DIR__, "..", "mms", "run_refinement.jl"))
