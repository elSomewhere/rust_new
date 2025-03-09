use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use log::debug;
use serde::{Serialize, Deserialize};
use glam::IVec3;
use lz4::block::compress;
use lz4::block::decompress;

use crate::voxel::CHUNK_SIZE;
use crate::voxel::types::VoxelData;
use crate::voxel::octree::Octree;
use crate::voxel::mesh::MeshStrategy;
use crate::worker::{WorkerSystem, WorkerTask};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkState {
    NotLoaded,
    Generating,
    Ready,
    Unloading,
    Compressed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub position: IVec3,
    pub state: ChunkState,
    pub voxel_data: Vec<VoxelData>,
    pub octree: Octree,
    pub mesh_data: HashMap<MeshStrategy, MeshData>,
    pub active_mesh_strategy: MeshStrategy,
    pub modified: bool,
    pub needs_remesh: bool,
    pub priority: f32,
    pub compressed_data: Option<Vec<u8>>,
}

impl Chunk {
    pub fn new(position: IVec3) -> Self {
        // Initialize an empty chunk
        let voxel_data = vec![VoxelData::air(); (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize];

        Self {
            position,
            state: ChunkState::NotLoaded,
            voxel_data,
            octree: Octree::new(),
            mesh_data: HashMap::new(),
            active_mesh_strategy: MeshStrategy::Instanced,
            modified: false,
            needs_remesh: false,
            priority: 0.0,
            compressed_data: None,
        }
    }

    pub fn get_voxel(&self, local_pos: IVec3) -> VoxelData {
        if self.is_valid_local_position(local_pos) {
            let index = self.get_voxel_index(local_pos);
            self.voxel_data[index]
        } else {
            VoxelData::air()
        }
    }

    pub fn set_voxel(&mut self, local_pos: IVec3, data: VoxelData) {
        if self.is_valid_local_position(local_pos) {
            let index = self.get_voxel_index(local_pos);
            self.voxel_data[index] = data;
            self.modified = true;
        }
    }

    fn get_voxel_index(&self, local_pos: IVec3) -> usize {
        (local_pos.x + local_pos.y * CHUNK_SIZE + local_pos.z * CHUNK_SIZE * CHUNK_SIZE) as usize
    }

    fn is_valid_local_position(&self, local_pos: IVec3) -> bool {
        local_pos.x >= 0 && local_pos.x < CHUNK_SIZE &&
            local_pos.y >= 0 && local_pos.y < CHUNK_SIZE &&
            local_pos.z >= 0 && local_pos.z < CHUNK_SIZE
    }

    pub fn update_octree(&mut self) {
        // Rebuild the entire octree based on the voxel data
        self.octree = Octree::new();

        // Build octree from voxel data
        self.octree.build_from_voxels(&self.voxel_data, CHUNK_SIZE);
    }

    pub fn update_octree_region(&mut self, local_pos: IVec3) {
        // Update octree for a specific region (after voxel modification)
        let region_size = 2; // Update a 2x2x2 region around the modified voxel

        let min_x = (local_pos.x - region_size).max(0);
        let min_y = (local_pos.y - region_size).max(0);
        let min_z = (local_pos.z - region_size).max(0);

        let max_x = (local_pos.x + region_size).min(CHUNK_SIZE - 1);
        let max_y = (local_pos.y + region_size).min(CHUNK_SIZE - 1);
        let max_z = (local_pos.z + region_size).min(CHUNK_SIZE - 1);

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    let update_pos = IVec3::new(x, y, z);
                    let voxel = self.get_voxel(update_pos);
                    self.octree.update_voxel(update_pos, voxel);
                }
            }
        }
    }

    pub fn mark_for_remesh(&mut self) {
        self.needs_remesh = true;
    }

    pub fn compress(&mut self) {
        if !self.modified {
            // Can be regenerated - just clear the data
            self.voxel_data = vec![];
            self.mesh_data.clear();
            self.state = ChunkState::NotLoaded;
            return;
        }

        // Serialize voxel data
        let voxel_bytes = bincode::serialize(&self.voxel_data).unwrap();

        // Compress with LZ4
        let compressed = compress(&voxel_bytes, None, false).unwrap();

        // Store compressed data
        self.compressed_data = Some(compressed);

        // Clear data to save memory
        self.voxel_data = vec![];
        self.mesh_data.clear();

        // Mark as compressed
        self.state = ChunkState::Compressed;
    }

    pub fn decompress(&mut self) {
        if let Some(ref compressed_data) = self.compressed_data {
            // Decompress voxel data
            let voxel_bytes = decompress(&compressed_data, None).unwrap();

            // Deserialize
            self.voxel_data = bincode::deserialize(&voxel_bytes).unwrap();

            // Update octree
            self.update_octree();

            // Clear compressed data
            self.compressed_data = None;

            // Mark as ready
            self.state = ChunkState::Ready;

            // Need to remesh
            self.mark_for_remesh();
        }
    }
}

// Import the MeshData from mesh.rs
// This is a workaround to avoid circular dependency
use crate::voxel::mesh::MeshData;

// Priority queue item for chunk loading
#[derive(Copy, Clone, Debug)]
struct ChunkPriority {
    position: IVec3,
    priority: f32,
}

impl PartialEq for ChunkPriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for ChunkPriority {}

impl PartialOrd for ChunkPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for ChunkPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// Chunk manager handles loading, unloading, and accessing chunks
pub struct ChunkManager {
    chunks: HashMap<IVec3, Chunk>,
    loading_queue: BinaryHeap<ChunkPriority>,
    unloading_queue: VecDeque<IVec3>,
    active_center: IVec3,
    active_distance: i32,
    default_mesh_strategy: MeshStrategy,
    pending_generations: HashMap<IVec3, bool>,
}

impl ChunkManager {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            loading_queue: BinaryHeap::new(),
            unloading_queue: VecDeque::new(),
            active_center: IVec3::new(0, 0, 0),
            active_distance: 8,
            default_mesh_strategy: MeshStrategy::Instanced,
            pending_generations: HashMap::new(),
        }
    }

    pub fn get_chunk(&self, position: IVec3) -> Option<&Chunk> {
        self.chunks.get(&position)
    }

    pub fn get_chunk_mut(&mut self, position: IVec3) -> Option<&mut Chunk> {
        self.chunks.get_mut(&position)
    }

    pub fn set_default_mesh_strategy(&mut self, strategy: MeshStrategy) {
        self.default_mesh_strategy = strategy;

        // Update active chunks
        for chunk in self.chunks.values_mut() {
            if chunk.state == ChunkState::Ready {
                chunk.active_mesh_strategy = strategy;
                if !chunk.mesh_data.contains_key(&strategy) {
                    chunk.mark_for_remesh();
                }
            }
        }
    }

    pub fn update(&mut self, center: IVec3, view_distance: i32, worker_system: &mut WorkerSystem) {
        self.active_center = center;
        self.active_distance = view_distance;

        // Update chunk priorities and decide what to load/unload
        self.refresh_chunk_priorities();

        // Schedule chunk loading/unloading
        self.process_chunk_queue(worker_system);
    }

    pub fn process_updates(&mut self, worker_system: &mut WorkerSystem, _default_strategy: MeshStrategy) {
        // First, collect all chunks that need remeshing and their positions
        let mut chunks_to_remesh = Vec::new();

        for (pos, chunk) in &self.chunks {
            if chunk.state == ChunkState::Ready && chunk.needs_remesh {
                chunks_to_remesh.push((*pos, chunk.active_mesh_strategy));
            }
        }

        // Now process each chunk that needs remeshing
        for (position, strategy) in chunks_to_remesh {
            // Mark as processed to avoid remeshing again until needed
            if let Some(chunk) = self.chunks.get_mut(&position) {
                chunk.needs_remesh = false;
            } else {
                continue; // Skip if chunk no longer exists
            }

            // Gather voxel data from this chunk and its neighbors
            let mut neighboring_chunks = HashMap::new();

            // Add this chunk's voxels
            if let Some(chunk) = self.chunks.get(&position) {
                neighboring_chunks.insert(position, chunk.voxel_data.clone());

                // Add all 26 neighboring chunks' voxels (including diagonal neighbors)
                // This is more comprehensive than just the 6 face-adjacent neighbors
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue; // Skip the center chunk (already added)
                            }

                            let neighbor_pos = position + IVec3::new(dx, dy, dz);
                            if let Some(neighbor) = self.chunks.get(&neighbor_pos) {
                                if neighbor.state == ChunkState::Ready {
                                    neighboring_chunks.insert(neighbor_pos, neighbor.voxel_data.clone());
                                }
                            }
                        }
                    }
                }

                // Submit the mesh generation task
                worker_system.submit_task(WorkerTask::GenerateMesh {
                    chunk_position: position,
                    strategy,
                    neighboring_chunks,
                });
            }
        }

        // Check for completed tasks
        worker_system.handle_completed_tasks(|task_result| {
            match task_result {
                crate::worker::WorkerTaskResult::ChunkGenerated {
                    position,
                    voxels,
                    octree,
                } => {
                    if let Some(chunk) = self.chunks.get_mut(&position) {
                        if chunk.state == ChunkState::Generating {
                            chunk.voxel_data = voxels;
                            chunk.octree = octree;
                            chunk.state = ChunkState::Ready;
                            chunk.active_mesh_strategy = self.default_mesh_strategy;
                            chunk.mark_for_remesh();

                            self.pending_generations.remove(&position);
                            debug!("Chunk at {:?} generation completed", position);
                        }
                    }
                },
                crate::worker::WorkerTaskResult::MeshGenerated {
                    position,
                    strategy,
                    mesh_data
                } => {
                    if let Some(chunk) = self.chunks.get_mut(&position) {
                        if chunk.state == ChunkState::Ready {
                            chunk.mesh_data.insert(strategy, mesh_data);
                            debug!("Mesh for chunk {:?} with strategy {:?} generated", position, strategy);
                        }
                    }
                },
                _ => {}
            }
        });
    }

    // Replace the refresh_chunk_priorities method in ChunkManager with this improved version
    fn refresh_chunk_priorities(&mut self) {
        // Clear the loading queue
        self.loading_queue.clear();

        // Calculate which chunks should be active
        let mut active_positions = Vec::new();
        let mut to_load = Vec::new();
        let mut to_unload = Vec::new();

        // Use a simple cubic distance for more uniform chunk loading
        let view_distance = self.active_distance;

        // First, get chunks in a cube around the player
        for x in -view_distance..=view_distance {
            for y in -view_distance..=view_distance {  // Full vertical range
                for z in -view_distance..=view_distance {
                    let offset = IVec3::new(x, y, z);

                    // Use simple cubic distance (max of absolute coordinates)
                    // This creates a more even distribution of chunks
                    let cubic_dist = offset.x.abs().max(offset.y.abs()).max(offset.z.abs());

                    // Check if within view distance using cubic distance
                    if cubic_dist <= view_distance {
                        let chunk_pos = self.active_center + offset;
                        active_positions.push(chunk_pos);

                        // Check if we need to load this chunk
                        if !self.chunks.contains_key(&chunk_pos) && !self.pending_generations.contains_key(&chunk_pos) {
                            // Base priority on distance from center (closer = higher priority)
                            // Use Manhattan distance for priority to ensure more even loading
                            let manhattan_dist = (offset.x.abs() + offset.y.abs() + offset.z.abs()) as f32;
                            let priority = 1000.0 - manhattan_dist;

                            to_load.push(ChunkPriority {
                                position: chunk_pos,
                                priority,
                            });
                        }
                    }
                }
            }
        }

        // Find chunks to unload
        for (pos, chunk) in &self.chunks {
            if chunk.state == ChunkState::Ready || chunk.state == ChunkState::Compressed {
                if !active_positions.contains(pos) {
                    to_unload.push(*pos);
                }
            }
        }

        // Add chunks to loading queue
        for chunk_priority in to_load {
            self.loading_queue.push(chunk_priority);
        }

        // Add chunks to unloading queue
        for pos in to_unload {
            if !self.unloading_queue.contains(&pos) {
                self.unloading_queue.push_back(pos);
            }
        }
    }

    fn process_chunk_queue(&mut self, worker_system: &mut WorkerSystem) {
        // Process a larger number of chunks per frame to ensure complete terrain loading
        const MAX_LOADS_PER_UPDATE: usize = 36;  // Increased from 12
        const MAX_UNLOADS_PER_UPDATE: usize = 12;  // Increased from 6

        // Process loading queue
        let mut loads_this_update = 0;
        while loads_this_update < MAX_LOADS_PER_UPDATE && !self.loading_queue.is_empty() {
            if let Some(chunk_priority) = self.loading_queue.pop() {
                let pos = chunk_priority.position;

                // If chunk isn't already being generated
                if !self.pending_generations.contains_key(&pos) {
                    // Create new chunk
                    let mut chunk = Chunk::new(pos);
                    chunk.state = ChunkState::Generating;
                    chunk.priority = chunk_priority.priority;

                    // Add to manager
                    self.chunks.insert(pos, chunk);

                    // Mark as pending
                    self.pending_generations.insert(pos, true);

                    // Submit generation task
                    worker_system.submit_task(WorkerTask::GenerateChunk { position: pos });

                    loads_this_update += 1;
                    debug!("Scheduled chunk generation for {:?}", pos);
                }
            }
        }

        // Process unloading queue
        let mut unloads_this_update = 0;
        while unloads_this_update < MAX_UNLOADS_PER_UPDATE && !self.unloading_queue.is_empty() {
            if let Some(pos) = self.unloading_queue.pop_front() {
                if let Some(chunk) = self.chunks.get_mut(&pos) {
                    if chunk.state == ChunkState::Ready {
                        // If modified, compress the chunk data
                        if chunk.modified {
                            chunk.compress();
                            debug!("Compressed chunk at {:?}", pos);
                        } else {
                            // Otherwise, just remove it (can be regenerated)
                            self.chunks.remove(&pos);
                            debug!("Removed unmodified chunk at {:?}", pos);
                        }

                        unloads_this_update += 1;
                    }
                }
            }
        }
    }

    pub fn get_world_bounds(&self) -> (IVec3, IVec3) {
        if self.chunks.is_empty() {
            return (IVec3::ZERO, IVec3::ZERO);
        }

        let mut min = IVec3::new(i32::MAX, i32::MAX, i32::MAX);
        let mut max = IVec3::new(i32::MIN, i32::MIN, i32::MIN);

        for pos in self.chunks.keys() {
            min.x = min.x.min(pos.x);
            min.y = min.y.min(pos.y);
            min.z = min.z.min(pos.z);

            max.x = max.x.max(pos.x);
            max.y = max.y.max(pos.y);
            max.z = max.z.max(pos.z);
        }

        (min, max)
    }

    pub fn get_active_chunks(&self) -> Vec<(IVec3, &Chunk)> {
        let mut active_chunks = Vec::new();

        for (pos, chunk) in &self.chunks {
            if chunk.state == ChunkState::Ready {
                active_chunks.push((*pos, chunk));
            }
        }

        active_chunks
    }
}