# High-Performance Voxel Engine

A high-performance voxel engine implementation in Rust and WebGPU, featuring multiple rendering strategies, infinite procedural terrain, and interactive world modification.

## Features

- **Instanced Rendering:** Efficiently renders large numbers of voxels using GPU instancing
- **Multiple Mesh Generation Strategies:**
    - Instanced: Direct instancing of template meshes (most memory-efficient)
    - Greedy Meshing: Merges adjacent faces to reduce vertex count (best for blocky terrain)
    - Marching Cubes: Generates smooth terrain from density fields (great for natural terrain)
    - Dual Contouring: Advanced contouring that preserves sharp features (for detailed surfaces)
- **Octree Spatial Partitioning:** Efficiently organizes voxels for fast spatial queries and culling
- **Chunk-based World Management:** Infinite, procedurally generated worlds
- **GPU-Accelerated:** Leverages compute shaders for culling and mesh generation
- **Multi-threaded:** Background worker system for generation and physics
- **Fully Destructible:** Modify terrain by adding or removing voxels

## Controls

- **WASD:** Move forward/backward/strafe left/right
- **Space/Ctrl:** Move up/down
- **Mouse:** Look around
- **Shift:** Sprint
- **Left Click:** Remove voxel
- **Space (hold):** Create explosion
- **1-4 Keys:** Switch between rendering strategies
    - 1: Instanced
    - 2: Greedy Meshing
    - 3: Marching Cubes
    - 4: Dual Contouring

## Building and Running

### Prerequisites

- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- Compatible graphics driver (Vulkan, Metal, or DX12)

### Build and Run

```sh
cargo build --release
cargo run --release
```

## Architecture Overview

The engine is built on three synergistic core components:

1. **Instanced Rendering:** Uses instancing to replicate a small set of voxel templates across the world
2. **Octree Spatial Partitioning:** Hierarchically divides the world space for efficient spatial queries
3. **Chunk-based World Management:** Organizes the world into fixed-size chunks for streaming and infinite worlds

### Key Technical Approaches

- **Hierarchical Octree Instancing:** Combines octree organization with instanced rendering
- **GPU-Driven Culling:** Offloads visibility determination to the GPU
- **Texture-Based Voxel Storage:** Stores voxel data in 3D textures for efficient GPU access
- **Asynchronous Architecture:** Worker threads for background tasks like procedural generation
- **Vertex Cache Optimization:** Optimizes meshes for GPU cache efficiency
- **Multiple Meshing Strategies:** Different approaches for different visual requirements

## Codebase Structure

- **src/main.rs:** Entry point and event handling
- **src/engine.rs:** Core engine implementation
- **src/voxel/:** Voxel representation and manipulation
    - **types.rs:** Voxel data structures
    - **chunk.rs:** Chunk management system
    - **octree.rs:** Octree implementation
    - **mesh.rs:** Mesh generation strategies
    - **procedural.rs:** Procedural terrain generation
- **src/rendering/:** Rendering system
    - **camera.rs:** Camera and view implementation
    - **pipeline.rs:** Rendering pipeline
    - **resources.rs:** GPU resources
    - **shaders.rs:** WGSL shader code
- **src/physics.rs:** Physics and collision system
- **src/worker.rs:** Async worker system for background tasks
- **src/utils.rs:** Utility functions and helpers

## License

MIT