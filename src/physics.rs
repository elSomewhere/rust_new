use std::collections::VecDeque;
use glam::{Vec3, IVec3};

use crate::voxel::World;
use crate::voxel::types::VoxelData;
use crate::utils::{world_to_chunk_coords, AABB};

pub struct PhysicsSystem {
    falling_blocks: Vec<FallingBlock>,
    fluid_updates: VecDeque<FluidUpdate>,
    max_updates_per_frame: usize,
}

struct FallingBlock {
    position: Vec3,
    velocity: Vec3,
    material: VoxelData,
    time_to_live: f32,
}

struct FluidUpdate {
    position: IVec3,
    fluid_type: u8,
    flow_level: u8,
}

impl PhysicsSystem {
    pub fn new() -> Self {
        Self {
            falling_blocks: Vec::new(),
            fluid_updates: VecDeque::new(),
            max_updates_per_frame: 1000,
        }
    }

    pub fn update(&mut self, world: &mut World, dt: f32) {
        // Update falling blocks
        self.update_falling_blocks(world, dt);

        // Process fluid updates
        self.update_fluids(world);
    }

    // Ray casting for voxel selection/destruction
    pub fn ray_cast(&self, world: &World, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<(IVec3, VoxelData)> {
        let dir = direction.normalize();

        // Step in small increments along the ray
        let step_size = 0.1;
        let max_steps = (max_distance / step_size) as i32;

        for i in 0..max_steps {
            let distance = i as f32 * step_size;
            let point = origin + dir * distance;

            // Convert to voxel coordinates
            let voxel_pos = IVec3::new(
                point.x.floor() as i32,
                point.y.floor() as i32,
                point.z.floor() as i32,
            );

            // Get voxel at this position
            let voxel = world.get_voxel(voxel_pos);

            // If not air, we hit something
            if !voxel.is_air() {
                return Some((voxel_pos, voxel));
            }
        }

        None
    }

    // Update falling blocks
    fn update_falling_blocks(&mut self, world: &mut World, dt: f32) {
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        // Update each falling block
        let mut i = 0;
        while i < self.falling_blocks.len() {
            let mut block = &mut self.falling_blocks[i];

            // Apply gravity
            block.velocity += gravity * dt;

            // Update position
            let new_position = block.position + block.velocity * dt;

            // Check for collision with existing voxels
            let voxel_pos = IVec3::new(
                new_position.x.floor() as i32,
                new_position.y.floor() as i32,
                new_position.z.floor() as i32,
            );

            let voxel_at_new_pos = world.get_voxel(voxel_pos);

            if voxel_at_new_pos.is_air() {
                // No collision, update position
                block.position = new_position;
                block.time_to_live -= dt;

                // If block has lived too long, remove it
                if block.time_to_live <= 0.0 {
                    world.modify_voxel(voxel_pos, block.material);
                    self.falling_blocks.swap_remove(i);
                    continue;
                }
            } else {
                // Collision, place the block at the last valid position
                let last_voxel_pos = IVec3::new(
                    block.position.x.floor() as i32,
                    block.position.y.floor() as i32,
                    block.position.z.floor() as i32,
                );

                if world.get_voxel(last_voxel_pos).is_air() {
                    world.modify_voxel(last_voxel_pos, block.material);
                }

                // Remove the falling block
                self.falling_blocks.swap_remove(i);
                continue;
            }

            i += 1;
        }
    }

    // Add a falling block
    pub fn add_falling_block(&mut self, position: Vec3, velocity: Vec3, material: VoxelData) {
        self.falling_blocks.push(FallingBlock {
            position,
            velocity,
            material,
            time_to_live: 5.0, // 5 seconds max fall time
        });
    }

    // Process fluid updates
    fn update_fluids(&mut self, world: &mut World) {
        // Process a limited number of fluid updates per frame
        let updates_this_frame = self.fluid_updates.len().min(self.max_updates_per_frame);

        for _ in 0..updates_this_frame {
            if let Some(update) = self.fluid_updates.pop_front() {
                self.process_fluid_update(world, update);
            }
        }
    }

    // Process a single fluid update
    fn process_fluid_update(&mut self, world: &mut World, update: FluidUpdate) {
        let pos = update.position;
        let fluid_type = update.fluid_type;
        let flow_level = update.flow_level;

        // Skip if at minimum flow level
        if flow_level == 0 {
            return;
        }

        // Flow downward first
        let below = pos + IVec3::new(0, -1, 0);
        if self.try_flow_fluid_to(world, below, fluid_type, flow_level) {
            // Flowed downward, no need to spread horizontally
            return;
        }

        // Then spread horizontally
        let new_flow = if flow_level > 1 { flow_level - 1 } else { 0 };

        // Try to flow in all four horizontal directions
        let directions = [
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(0, 0, -1),
        ];

        for dir in &directions {
            self.try_flow_fluid_to(world, pos + *dir, fluid_type, new_flow);
        }
    }

    // Try to flow fluid to a position
    fn try_flow_fluid_to(&mut self, world: &mut World, pos: IVec3, fluid_type: u8, flow_level: u8) -> bool {
        // Get current voxel at position
        let current = world.get_voxel(pos);

        // If it's air, we can flow there
        if current.is_air() {
            // Create fluid voxel with the specified flow level
            let fluid = if fluid_type == 5 {
                VoxelData::water()
            } else {
                VoxelData::air() // Default to air for unknown fluids
            };

            // Set the fluid
            world.modify_voxel(pos, fluid);

            // Add a new fluid update for this position
            self.fluid_updates.push_back(FluidUpdate {
                position: pos,
                fluid_type,
                flow_level,
            });

            return true;
        }
        // If it's already the same fluid with a lower flow level, update it
        else if current.material_id == fluid_type && current.density < flow_level as u8 {
            let mut new_fluid = current;
            new_fluid.density = flow_level as u8;

            // Update the fluid
            world.modify_voxel(pos, new_fluid);

            // Add a new fluid update for this position
            self.fluid_updates.push_back(FluidUpdate {
                position: pos,
                fluid_type,
                flow_level,
            });

            return true;
        }

        false
    }

    // Add a fluid source
    pub fn add_fluid_source(&mut self, position: IVec3, fluid_type: u8) {
        self.fluid_updates.push_back(FluidUpdate {
            position,
            fluid_type,
            flow_level: 7, // Maximum flow level
        });
    }

    // Check for collisions between an AABB and the voxel world
    pub fn check_collision(&self, world: &World, aabb: &AABB) -> bool {
        // Get voxel coordinates for the AABB
        let min_voxel = IVec3::new(
            aabb.min.x.floor() as i32,
            aabb.min.y.floor() as i32,
            aabb.min.z.floor() as i32,
        );

        let max_voxel = IVec3::new(
            aabb.max.x.floor() as i32,
            aabb.max.y.floor() as i32,
            aabb.max.z.floor() as i32,
        );

        // Check each voxel in the AABB
        for x in min_voxel.x..=max_voxel.x {
            for y in min_voxel.y..=max_voxel.y {
                for z in min_voxel.z..=max_voxel.z {
                    let voxel_pos = IVec3::new(x, y, z);
                    let voxel = world.get_voxel(voxel_pos);

                    if voxel.is_solid() {
                        // Create AABB for this voxel
                        let voxel_aabb = AABB::new(
                            Vec3::new(x as f32, y as f32, z as f32),
                            Vec3::new((x + 1) as f32, (y + 1) as f32, (z + 1) as f32),
                        );

                        // Check for intersection
                        if aabb.intersects(&voxel_aabb) {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    // Find the closest point on the voxel grid surface to a ray
    pub fn closest_surface_point(&self, world: &World, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<(IVec3, Vec3)> {
        if let Some((voxel_pos, _)) = self.ray_cast(world, origin, direction, max_distance) {
            // Get the exact intersection point
            let t = self.ray_box_intersection(
                origin,
                direction,
                Vec3::new(voxel_pos.x as f32, voxel_pos.y as f32, voxel_pos.z as f32),
                Vec3::new((voxel_pos.x + 1) as f32, (voxel_pos.y + 1) as f32, (voxel_pos.z + 1) as f32),
            )?;

            let intersection = origin + direction * t;

            return Some((voxel_pos, intersection));
        }

        None
    }

    // Helper function to calculate ray-box intersection
    fn ray_box_intersection(&self, origin: Vec3, direction: Vec3, box_min: Vec3, box_max: Vec3) -> Option<f32> {
        let dir_inv = Vec3::new(
            1.0 / direction.x,
            1.0 / direction.y,
            1.0 / direction.z,
        );

        let t1 = (box_min.x - origin.x) * dir_inv.x;
        let t2 = (box_max.x - origin.x) * dir_inv.x;
        let t3 = (box_min.y - origin.y) * dir_inv.y;
        let t4 = (box_max.y - origin.y) * dir_inv.y;
        let t5 = (box_min.z - origin.z) * dir_inv.z;
        let t6 = (box_max.z - origin.z) * dir_inv.z;

        let tmin = f32::max(
            f32::max(f32::min(t1, t2), f32::min(t3, t4)),
            f32::min(t5, t6),
        );
        let tmax = f32::min(
            f32::min(f32::max(t1, t2), f32::max(t3, t4)),
            f32::max(t5, t6),
        );

        // Ray is intersecting AABB, but whole AABB is behind us
        if tmax < 0.0 {
            return None;
        }

        // Ray doesn't intersect AABB
        if tmin > tmax {
            return None;
        }

        Some(if tmin < 0.0 { 0.0 } else { tmin })
    }
}