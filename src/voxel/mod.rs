use std::sync::{Arc, RwLock};
use glam::{Vec3, IVec3};
use log::info;

pub mod types;
pub mod chunk;
pub mod octree;
pub mod mesh;
pub mod procedural;

use types::VoxelData;
use chunk::{ChunkManager, Chunk, ChunkState};
use octree::Octree;
use mesh::MeshStrategy;
use crate::rendering::Camera;
use crate::worker::WorkerSystem;
use crate::utils::{AABB, world_to_chunk_coords};

pub const CHUNK_SIZE: i32 = 16;
pub const VIEW_DISTANCE: i32 = 8;

pub struct World {
    chunks: Arc<RwLock<ChunkManager>>,
    mesh_strategy: MeshStrategy,
}

impl World {
    pub fn new() -> Self {
        let chunks = Arc::new(RwLock::new(ChunkManager::new()));

        Self {
            chunks,
            mesh_strategy: MeshStrategy::Instanced,
        }
    }

    pub fn get_chunks(&self) -> Arc<RwLock<ChunkManager>> {
        Arc::clone(&self.chunks)
    }

    pub fn set_mesh_strategy(&mut self, strategy: MeshStrategy) {
        self.mesh_strategy = strategy;
        self.chunks.write().unwrap().set_default_mesh_strategy(strategy);
    }

    pub fn get_voxel(&self, pos: IVec3) -> VoxelData {
        let (chunk_pos, local_pos) = world_to_chunk_coords(pos, CHUNK_SIZE);

        let chunks = self.chunks.read().unwrap();
        if let Some(chunk) = chunks.get_chunk(chunk_pos) {
            if chunk.state == ChunkState::Ready {
                return chunk.get_voxel(local_pos);
            }
        }

        // Default to air if chunk not loaded
        VoxelData::air()
    }

    pub fn modify_voxel(&self, pos: IVec3, data: VoxelData) {
        let (chunk_pos, local_pos) = world_to_chunk_coords(pos, CHUNK_SIZE);

        let mut chunks = self.chunks.write().unwrap();
        if let Some(chunk) = chunks.get_chunk_mut(chunk_pos) {
            if chunk.state == ChunkState::Ready {
                chunk.set_voxel(local_pos, data);
                chunk.modified = true;

                // Update the region in the octree
                chunk.update_octree_region(local_pos);

                // Mark for remeshing
                chunk.mark_for_remesh();
            }
        }
    }

    pub fn create_explosion(&self, center: IVec3, radius: i32) {
        let r_squared = radius * radius;

        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let offset = IVec3::new(x, y, z);
                    let dist_squared = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;

                    if dist_squared <= r_squared {
                        self.modify_voxel(center + offset, VoxelData::air());
                    }
                }
            }
        }
    }

    pub fn update(&mut self, player_chunk_pos: IVec3, camera: &Camera, worker_system: &mut WorkerSystem) {
        // Update chunk loading/unloading based on player position
        let mut chunks = self.chunks.write().unwrap();
        chunks.update(player_chunk_pos, VIEW_DISTANCE, worker_system);

        // Process chunk updates and mesh regeneration
        chunks.process_updates(worker_system, self.mesh_strategy);
    }

    pub fn get_aabb(&self) -> AABB {
        let chunks = self.chunks.read().unwrap();
        let (min, max) = chunks.get_world_bounds();

        AABB::new(
            Vec3::new(
                min.x as f32 * CHUNK_SIZE as f32,
                min.y as f32 * CHUNK_SIZE as f32,
                min.z as f32 * CHUNK_SIZE as f32,
            ),
            Vec3::new(
                (max.x + 1) as f32 * CHUNK_SIZE as f32,
                (max.y + 1) as f32 * CHUNK_SIZE as f32,
                (max.z + 1) as f32 * CHUNK_SIZE as f32,
            ),
        )
    }
}