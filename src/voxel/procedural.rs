use glam::IVec3;
use noise::{NoiseFn, Perlin, Fbm, MultiFractal, Seedable};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;

use crate::voxel::types::VoxelData;
use crate::voxel::octree::Octree;
use crate::voxel::{CHUNK_SIZE};

pub struct TerrainGenerator {
    seed: u32,
    height_noise: Fbm<Perlin>,
    cave_noise: Fbm<Perlin>,
    detail_noise: Perlin,
    biome_noise: Fbm<Perlin>,
}

impl TerrainGenerator {
    pub fn new(seed: u32) -> Self {
        // Create noise generators with different seeds
        let height_noise = Fbm::<Perlin>::new(seed)
            .set_octaves(6)
            .set_frequency(0.005)
            .set_persistence(0.5)
            .set_lacunarity(2.0);

        let cave_noise = Fbm::<Perlin>::new(seed.wrapping_add(123))
            .set_octaves(4)
            .set_frequency(0.07)
            .set_persistence(0.5)
            .set_lacunarity(2.0);

        let detail_noise = Perlin::new(seed.wrapping_add(456));

        let biome_noise = Fbm::<Perlin>::new(seed.wrapping_add(789))
            .set_octaves(3)
            .set_frequency(0.002)
            .set_persistence(0.5)
            .set_lacunarity(2.0);

        Self {
            seed,
            height_noise,
            cave_noise,
            detail_noise,
            biome_noise,
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

                // Generate terrain height at this position
                let height = self.generate_height(world_x as f64, world_z as f64);

                // Adjust height to world coordinates
                let world_height = height as i32;

                for y in 0..CHUNK_SIZE {
                    let world_y = chunk_pos.y * CHUNK_SIZE + y;

                    // Check if we're at or below the surface
                    if world_y <= world_height {
                        // Get voxel index
                        let voxel_index = (x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE) as usize;

                        // Determine voxel type based on depth
                        let voxel = if world_y == world_height {
                            // Surface - generate proper block based on biome
                            self.get_surface_voxel(world_x, world_y, world_z)
                        } else if world_y > world_height - 3 {
                            // Near surface - dirt
                            VoxelData::dirt()
                        } else {
                            // Deep underground - stone
                            VoxelData::stone()
                        };

                        // Check for caves
                        let cave_density = self.generate_cave_density(world_x as f64, world_y as f64, world_z as f64);
                        if cave_density > 0.75 {  // Increased from 0.58 to 0.75
                            voxels[voxel_index] = voxel;
                        } else {
                            voxels[voxel_index] = VoxelData::air();
                        }
                    }

                    // Check for water (anything below sea level)
                    let sea_level = 60;
                    let voxel_index = (x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE) as usize;

                    if world_y < sea_level && voxels[voxel_index].is_air() {
                        voxels[voxel_index] = VoxelData::water();
                    }
                }
            }
        }

        // Build octree
        let mut octree = Octree::new();
        octree.build_from_voxels(&voxels, CHUNK_SIZE);

        (voxels, octree)
    }

    fn generate_height(&self, x: f64, z: f64) -> f64 {
        // Base continent shape
        let base_height = self.height_noise.get([x, z]) * 80.0 + 60.0;

        // Add some detail variation
        let detail = self.detail_noise.get([x * 0.2, z * 0.2]) * 5.0;

        base_height + detail
    }

    fn generate_cave_density(&self, x: f64, y: f64, z: f64) -> f64 {
        // 3D noise for caves
        // Increase the threshold to make caves less common (0.3 to 0.5)
        self.cave_noise.get([x * 0.05, y * 0.05, z * 0.05]) + 0.5
    }

    fn get_surface_voxel(&self, x: i32, y: i32, z: i32) -> VoxelData {
        // Generate biome value from 0.0 to 1.0
        let biome_value = (self.biome_noise.get([x as f64 * 0.01, z as f64 * 0.01]) + 1.0) / 2.0;

        // Get temperature/moisture from biome value (simplified model)
        let temperature = (biome_value * 255.0) as u8;
        let moisture = (self.biome_noise.get([x as f64 * 0.01 + 500.0, z as f64 * 0.01 + 500.0]) * 127.0 + 128.0) as u8;

        // Determine block type based on temperature/moisture
        if y < 60 {
            // Below sea level - sand near water
            VoxelData::sand()
        } else if temperature > 200 && moisture < 100 {
            // Hot and dry - desert
            VoxelData::sand()
        } else if temperature < 100 && moisture > 150 {
            // Cold and wet - stone
            VoxelData::stone()
        } else {
            // Moderate conditions - grass
            VoxelData::grass()
        }
    }

    // Generate various structures
    pub fn generate_structure(&self, pos: IVec3, voxels: &mut Vec<VoxelData>, structure_type: StructureType) {
        match structure_type {
            StructureType::Tree => self.generate_tree(pos, voxels),
            StructureType::Cave => self.generate_cave(pos, voxels),
            StructureType::Ore => self.generate_ore_vein(pos, voxels),
        }
    }

    fn generate_tree(&self, pos: IVec3, voxels: &mut Vec<VoxelData>) {
        // Simplified tree generation
        let trunk_height = 4 + (self.detail_noise.get([pos.x as f64 * 0.1, pos.z as f64 * 0.1]) * 3.0) as i32;

        // Check bounds to avoid buffer overflows
        let max_height = CHUNK_SIZE - 1;
        let effective_height = trunk_height.min(max_height - pos.y);

        // Generate trunk
        for y in 0..effective_height {
            let index = self.get_voxel_index(IVec3::new(pos.x, pos.y + y, pos.z));
            if index < voxels.len() {
                voxels[index] = VoxelData::new(6, 255, 128, 128); // Wood
            }
        }

        // Generate leaves
        let leaf_radius = 2;
        for y in (effective_height - 3).max(0)..=effective_height {
            for x in -leaf_radius..=leaf_radius {
                for z in -leaf_radius..=leaf_radius {
                    // Skip trunk
                    if x == 0 && z == 0 && y < effective_height {
                        continue;
                    }

                    // Simple sphere-ish shape for leaves
                    let dist = (x * x + (y - effective_height) * (y - effective_height) + z * z) as f32;
                    if dist <= (leaf_radius * leaf_radius) as f32 {
                        let leaf_pos = IVec3::new(pos.x + x, pos.y + y, pos.z + z);
                        let index = self.get_voxel_index(leaf_pos);
                        if index < voxels.len() {
                            voxels[index] = VoxelData::new(7, 200, 128, 200); // Leaves
                        }
                    }
                }
            }
        }
    }

    fn generate_cave(&self, pos: IVec3, voxels: &mut Vec<VoxelData>) {
        // Simple spherical cave
        let radius = 3 + (self.detail_noise.get([pos.x as f64 * 0.1, pos.z as f64 * 0.1]) * 2.0) as i32;

        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let dist_squared = x*x + y*y + z*z;
                    if dist_squared <= radius * radius {
                        let cave_pos = IVec3::new(pos.x + x, pos.y + y, pos.z + z);
                        let index = self.get_voxel_index(cave_pos);
                        if index < voxels.len() {
                            voxels[index] = VoxelData::air();
                        }
                    }
                }
            }
        }
    }

    fn generate_ore_vein(&self, pos: IVec3, voxels: &mut Vec<VoxelData>) {
        // Simple ore vein generation
        let radius = 2 + (self.detail_noise.get([pos.x as f64 * 0.3, pos.z as f64 * 0.3]) * 1.5) as i32;

        // Determine ore type based on depth
        let ore_material = if pos.y < 30 {
            8 // Gold
        } else if pos.y < 50 {
            9 // Iron
        } else {
            10 // Coal
        };

        let mut rng = Pcg32::seed_from_u64((pos.x as u64) << 32 | (pos.y as u64) << 16 | pos.z as u64);

        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let dist_squared = x*x + y*y + z*z;
                    if dist_squared <= radius * radius {
                        // Add some randomness to make it less spherical
                        if rng.gen::<f32>() < 0.7 {
                            let ore_pos = IVec3::new(pos.x + x, pos.y + y, pos.z + z);
                            let index = self.get_voxel_index(ore_pos);
                            if index < voxels.len() && voxels[index].material_id == 1 { // Only replace stone
                                voxels[index] = VoxelData::new(ore_material, 255, 128, 128);
                            }
                        }
                    }
                }
            }
        }
    }

    fn get_voxel_index(&self, pos: IVec3) -> usize {
        let x = pos.x.rem_euclid(CHUNK_SIZE);
        let y = pos.y.rem_euclid(CHUNK_SIZE);
        let z = pos.z.rem_euclid(CHUNK_SIZE);

        (x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE) as usize
    }
}

pub enum StructureType {
    Tree,
    Cave,
    Ore,
}

// Optional wave intrinsics implementation with feature flag
#[cfg(feature = "wave_intrinsics")]
pub mod wave_optimized {
    use super::*;

    pub struct WaveOptimizedTerrainGenerator {
        inner: TerrainGenerator,
    }

    impl WaveOptimizedTerrainGenerator {
        pub fn new(seed: u32) -> Self {
            Self {
                inner: TerrainGenerator::new(seed),
            }
        }

        // This function would use wave intrinsics in the compute shader
        // Here we just define the interface
        pub fn generate_chunk_wave_optimized(&self, chunk_pos: IVec3) -> (Vec<VoxelData>, Octree) {
            // In real implementation, this would generate a compute shader with wave intrinsics
            // For now, fall back to regular implementation
            self.inner.generate_chunk(chunk_pos)
        }
    }
}