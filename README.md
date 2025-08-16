# DAGE-Dynamically-Attributed-Grammatical-Evolution-

## Overview

Dynamically Attributed Grammatical Evolution (DAGE) is a Python framework for Grammatical Evolution (GE) that extends traditional Context-Free Grammar (CFG) approaches by integrating dynamic semantic checks directly into the grammar. Unlike conventional Attribute Grammars (AGs), which are static and often complex, DAGE allows functions to be used as non-terminals. These functions are invoked during genotype-to-phenotype mapping, enabling runtime-aware mapping, dynamic context-sensitive behavior, and direct interaction with external systems.

By embedding semantic operations into the grammar itself, DAGE simplifies the enforcement of constraints, increases expressivity, and facilitates the design of adaptable, real-world-ready evolutionary systems. The framework is built on Python and leverages the GRAPE and DEAP libraries, making it accessible to researchers and practitioners alike.

## Features

- Dynamic, function-based grammar non-terminals for runtime-aware mapping.

- On-the-fly semantic checks and constraint enforcement.

- Simplified grammar design through reusable functions.

- Compatibility with external systems during evolutionary runs.

- Benchmark-tested for flexibility across diverse problem domains.

## Supported Benchmarks

DAGE has been evaluated on several benchmark problems, demonstrating its versatility:

- Division Problem (SRP)

- Multi-Type Knapsack Problem (MKP)

- Artificial Ant Problem (AAP)

These experiments show that DAGE can easily adapt to any problem requiring semantic checking.
