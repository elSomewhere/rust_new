use glam::IVec3;
use noise::{NoiseFn, Perlin};

use crate::voxel::types::VoxelData;
use crate::voxel::octree::Octree;
use crate::voxel::CHUNK_SIZE;

pub struct SimpleTerrainGenerator {
    seed: u32,
    noise: Perlin,
}

impl SimpleTerrainGenerator {
    pub fn new(seed: u32) -> Self {
        // Create simple noise generator
        let noise = Perlin::new(seed);

        Self {
            seed,
            noise,
        }
    }

    pub fn generate_chunk(&self, chunk_pos: IVec3) -> (Vec<VoxelData>, Octree) {
        let mut voxels = vec![VoxelData::air(); (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize];

        // Generate the chunk
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                // Calculate world coordinates
                let world_x = chunk_pos.x * CHUNK_SIZE + x;
                let world_z = chunk_pos.z * CHUNK_SIZE + z;

                // Generate terrain height using a single simplified noise function
                // Lower frequency (0.005) creates more gradual terrain
                let base_height = 40.0; // Base terrain height
                let amplitude = 20.0;   // Height variation

                let noise_val = self.noise.get([world_x as f64 * 0.005, world_z as f64 * 0.005]);
                let height = (base_height + noise_val * amplitude) as i32;

                for y in 0..CHUNK_SIZE {
                    let world_y = chunk_pos.y * CHUNK_SIZE + y;

                    // Check if we're at or below the surface
                    if world_y <= height {
                        // Get voxel index
                        let voxel_index = (x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE) as usize;

                        // Just use a single standard tile type for everything
                        voxels[voxel_index] = VoxelData::stone();
                    }
                }
            }
        }

        // Build octree
        let mut octree = Octree::new();
        octree.build_from_voxels(&voxels, CHUNK_SIZE);

        (voxels, octree)
    }
}

// Add a function to replace the existing TerrainGenerator
pub fn create_simple_terrain_generator(seed: u32) -> SimpleTerrainGenerator {
    SimpleTerrainGenerator::new(seed)
}