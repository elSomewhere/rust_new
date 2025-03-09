use std::ops::Range;
use glam::{IVec3, Vec3};

// Range utils
pub fn for_3d_range<F>(range: Range<i32>, mut f: F)
where
    F: FnMut(i32, i32, i32),
{
    for x in range.clone() {
        for y in range.clone() {
            for z in range.clone() {
                f(x, y, z);
            }
        }
    }
}

// Z-order curve functions
pub fn morton_encode(x: u32, y: u32, z: u32) -> u32 {
    let mut result = 0u32;
    for i in 0..10 {
        let x_bit = (x >> i) & 1;
        let y_bit = (y >> i) & 1;
        let z_bit = (z >> i) & 1;
        result |= (x_bit << (3 * i)) | (y_bit << (3 * i + 1)) | (z_bit << (3 * i + 2));
    }
    result
}

pub fn morton_decode(code: u32) -> (u32, u32, u32) {
    let mut x = 0u32;
    let mut y = 0u32;
    let mut z = 0u32;

    for i in 0..10 {
        x |= ((code >> (3 * i)) & 1) << i;
        y |= ((code >> (3 * i + 1)) & 1) << i;
        z |= ((code >> (3 * i + 2)) & 1) << i;
    }

    (x, y, z)
}

// Direction and position utilities
pub const DIRECTIONS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

pub const ALL_NEIGHBORS: [IVec3; 26] = [
    // Same Y level (8)
    IVec3::new(-1, 0, -1), IVec3::new(0, 0, -1), IVec3::new(1, 0, -1),
    IVec3::new(-1, 0, 0),                        IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 1),  IVec3::new(0, 0, 1),  IVec3::new(1, 0, 1),

    // Below (9)
    IVec3::new(-1, -1, -1), IVec3::new(0, -1, -1), IVec3::new(1, -1, -1),
    IVec3::new(-1, -1, 0),  IVec3::new(0, -1, 0),  IVec3::new(1, -1, 0),
    IVec3::new(-1, -1, 1),  IVec3::new(0, -1, 1),  IVec3::new(1, -1, 1),

    // Above (9)
    IVec3::new(-1, 1, -1), IVec3::new(0, 1, -1), IVec3::new(1, 1, -1),
    IVec3::new(-1, 1, 0),  IVec3::new(0, 1, 0),  IVec3::new(1, 1, 0),
    IVec3::new(-1, 1, 1),  IVec3::new(0, 1, 1),  IVec3::new(1, 1, 1),
];

// Convert between coordinate systems
// In utils.rs, ensure the world_to_chunk_coords function is correct:
// Convert between coordinate systems
pub fn world_to_chunk_coords(world_pos: IVec3, chunk_size: i32) -> (IVec3, IVec3) {
    // Use integer division for chunk position (flooring division for negative numbers)
    let chunk_pos = IVec3::new(
        world_pos.x.div_euclid(chunk_size),
        world_pos.y.div_euclid(chunk_size),
        world_pos.z.div_euclid(chunk_size),
    );

    // Get local position within chunk (always positive)
    let local_pos = IVec3::new(
        world_pos.x.rem_euclid(chunk_size),
        world_pos.y.rem_euclid(chunk_size),
        world_pos.z.rem_euclid(chunk_size),
    );

    (chunk_pos, local_pos)
}

pub fn chunk_to_world_coords(chunk_pos: IVec3, local_pos: IVec3, chunk_size: i32) -> IVec3 {
    chunk_pos * chunk_size + local_pos
}

// AABB utilities
#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_center_half_size(center: Vec3, half_size: Vec3) -> Self {
        Self {
            min: center - half_size,
            max: center + half_size,
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_size(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
            point.y >= self.min.y && point.y <= self.max.y &&
            point.z >= self.min.z && point.z <= self.max.z
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
            self.min.y <= other.max.y && self.max.y >= other.min.y &&
            self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    pub fn ray_intersects(&self, origin: Vec3, direction: Vec3, max_dist: f32) -> Option<f32> {
        let inv_dir = Vec3::new(
            1.0 / direction.x,
            1.0 / direction.y,
            1.0 / direction.z,
        );

        let t1 = (self.min.x - origin.x) * inv_dir.x;
        let t2 = (self.max.x - origin.x) * inv_dir.x;
        let t3 = (self.min.y - origin.y) * inv_dir.y;
        let t4 = (self.max.y - origin.y) * inv_dir.y;
        let t5 = (self.min.z - origin.z) * inv_dir.z;
        let t6 = (self.max.z - origin.z) * inv_dir.z;

        let tmin = f32::max(
            f32::max(f32::min(t1, t2), f32::min(t3, t4)),
            f32::min(t5, t6),
        );
        let tmax = f32::min(
            f32::min(f32::max(t1, t2), f32::max(t3, t4)),
            f32::max(t5, t6),
        );

        if tmax < 0.0 || tmin > tmax || tmin > max_dist {
            None
        } else {
            Some(if tmin < 0.0 { 0.0 } else { tmin })
        }
    }
}

// Frustum utilities
#[derive(Debug)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

#[derive(Clone, Copy, Debug)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn dot(&self, v: Vec3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z + self.w
    }

    pub fn normalize(&mut self) {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0 {
            let inv_len = 1.0 / len;
            self.x *= inv_len;
            self.y *= inv_len;
            self.z *= inv_len;
            self.w *= inv_len;
        }
    }
}

impl Frustum {
    pub fn from_view_projection(view_proj: &glam::Mat4) -> Self {
        let mut planes = [Vec4::new(0.0, 0.0, 0.0, 0.0); 6];

        // Extract planes from view-projection matrix
        // Left plane
        planes[0] = Vec4::new(
            view_proj.col(3).x + view_proj.col(0).x,
            view_proj.col(3).y + view_proj.col(0).y,
            view_proj.col(3).z + view_proj.col(0).z,
            view_proj.col(3).w + view_proj.col(0).w,
        );

        // Right plane
        planes[1] = Vec4::new(
            view_proj.col(3).x - view_proj.col(0).x,
            view_proj.col(3).y - view_proj.col(0).y,
            view_proj.col(3).z - view_proj.col(0).z,
            view_proj.col(3).w - view_proj.col(0).w,
        );

        // Bottom plane
        planes[2] = Vec4::new(
            view_proj.col(3).x + view_proj.col(1).x,
            view_proj.col(3).y + view_proj.col(1).y,
            view_proj.col(3).z + view_proj.col(1).z,
            view_proj.col(3).w + view_proj.col(1).w,
        );

        // Top plane
        planes[3] = Vec4::new(
            view_proj.col(3).x - view_proj.col(1).x,
            view_proj.col(3).y - view_proj.col(1).y,
            view_proj.col(3).z - view_proj.col(1).z,
            view_proj.col(3).w - view_proj.col(1).w,
        );

        // Near plane
        planes[4] = Vec4::new(
            view_proj.col(2).x,
            view_proj.col(2).y,
            view_proj.col(2).z,
            view_proj.col(2).w,
        );

        // Far plane
        planes[5] = Vec4::new(
            view_proj.col(3).x - view_proj.col(2).x,
            view_proj.col(3).y - view_proj.col(2).y,
            view_proj.col(3).z - view_proj.col(2).z,
            view_proj.col(3).w - view_proj.col(2).w,
        );

        // Normalize planes
        for plane in &mut planes {
            plane.normalize();
        }

        Self { planes }
    }

    pub fn contains_point(&self, point: Vec3) -> bool {
        for plane in &self.planes {
            if plane.dot(point) < 0.0 {
                return false;
            }
        }
        true
    }

    pub fn contains_aabb(&self, aabb: &AABB) -> bool {
        let center = aabb.center();
        let half_size = aabb.half_size();

        for plane in &self.planes {
            let p = Vec3::new(
                if plane.x > 0.0 { half_size.x } else { -half_size.x },
                if plane.y > 0.0 { half_size.y } else { -half_size.y },
                if plane.z > 0.0 { half_size.z } else { -half_size.z },
            );

            if plane.dot(center + p) < 0.0 {
                return false;
            }
        }

        true
    }
}

// Vertex cache optimization
pub struct VertexCacheOptimizer {
    cache_size: usize,
    cache: Vec<i32>,
    score_table: Vec<f32>,
}

impl VertexCacheOptimizer {
    pub fn new(cache_size: usize) -> Self {
        let mut score_table = vec![0.0; 64];

        // Precompute scores
        for i in 0..64 {
            let score = if i < cache_size {
                // Direct cache scores
                (cache_size as f32 - i as f32) / cache_size as f32
            } else {
                // Score for vertices not in cache
                0.0
            };
            score_table[i] = score;
        }

        Self {
            cache_size,
            cache: vec![-1; cache_size],
            score_table,
        }
    }

    pub fn calculate_vertex_score(&self, cache_position: i32, remaining_valence: i32) -> f32 {
        // Score for vertices not in cache
        if cache_position < 0 {
            // Bonus points for having low valence
            return 0.75 * (if remaining_valence <= 2 { 1.0 } else { (2.0 * (remaining_valence as f32).powf(-0.5)) })
        }

        // Score for vertices in cache
        self.score_table[cache_position as usize] * (if remaining_valence <= 2 { 1.0 } else { 0.5 })
    }

    pub fn optimize(&mut self, indices: &mut [u32], vertex_count: usize) {
        if indices.len() < 3 || vertex_count == 0 {
            return;
        }

        // Calculate vertex valence (number of triangles a vertex is part of)
        let mut valence = vec![0; vertex_count];
        for i in (0..indices.len()).step_by(3) {
            valence[indices[i] as usize] += 1;
            valence[indices[i + 1] as usize] += 1;
            valence[indices[i + 2] as usize] += 1;
        }

        // Working values
        let mut remaining_valence = valence.clone();
        let mut live_triangles = vec![true; indices.len() / 3];
        let mut live_triangle_count = live_triangles.len();

        // Output triangle list
        let mut new_indices = Vec::with_capacity(indices.len());

        // Reset cache
        self.cache.fill(-1);

        // Main optimization loop
        while live_triangle_count > 0 {
            // Find best triangle
            let mut best_score = -1.0;
            let mut best_triangle = -1;

            for i in 0..live_triangles.len() {
                if !live_triangles[i] {
                    continue;
                }

                // Get vertices for this triangle
                let v0 = indices[i * 3] as usize;
                let v1 = indices[i * 3 + 1] as usize;
                let v2 = indices[i * 3 + 2] as usize;

                // Find cache positions
                let p0 = self.cache.iter().position(|&x| x == v0 as i32).map_or(-1, |p| p as i32);
                let p1 = self.cache.iter().position(|&x| x == v1 as i32).map_or(-1, |p| p as i32);
                let p2 = self.cache.iter().position(|&x| x == v2 as i32).map_or(-1, |p| p as i32);

                // Calculate score
                let score = self.calculate_vertex_score(p0, remaining_valence[v0]) +
                    self.calculate_vertex_score(p1, remaining_valence[v1]) +
                    self.calculate_vertex_score(p2, remaining_valence[v2]);

                if score > best_score {
                    best_score = score;
                    best_triangle = i as i32;
                }
            }

            if best_triangle == -1 {
                break; // Something went wrong
            }

            // Add best triangle to output
            let tri_idx = best_triangle as usize * 3;
            new_indices.push(indices[tri_idx]);
            new_indices.push(indices[tri_idx + 1]);
            new_indices.push(indices[tri_idx + 2]);

            // Mark as used
            live_triangles[best_triangle as usize] = false;
            live_triangle_count -= 1;

            // Update remaining valence
            let v0 = indices[tri_idx] as usize;
            let v1 = indices[tri_idx + 1] as usize;
            let v2 = indices[tri_idx + 2] as usize;

            remaining_valence[v0] -= 1;
            remaining_valence[v1] -= 1;
            remaining_valence[v2] -= 1;

            // Update vertex cache
            // Remove vertices that are now done
            if remaining_valence[v0] == 0 {
                if let Some(pos) = self.cache.iter().position(|&x| x == v0 as i32) {
                    self.cache[pos] = -1;
                }
            }

            if remaining_valence[v1] == 0 {
                if let Some(pos) = self.cache.iter().position(|&x| x == v1 as i32) {
                    self.cache[pos] = -1;
                }
            }

            if remaining_valence[v2] == 0 {
                if let Some(pos) = self.cache.iter().position(|&x| x == v2 as i32) {
                    self.cache[pos] = -1;
                }
            }

            // Add triangle vertices to cache (LRU style)
            self.add_to_cache(v0 as i32);
            self.add_to_cache(v1 as i32);
            self.add_to_cache(v2 as i32);
        }

        // Replace indices with optimized ones
        indices.copy_from_slice(&new_indices);
    }

    fn add_to_cache(&mut self, vertex: i32) {
        // Check if already in cache
        if let Some(pos) = self.cache.iter().position(|&x| x == vertex) {
            // Move to front
            for i in (1..=pos).rev() {
                self.cache[i] = self.cache[i - 1];
            }
            self.cache[0] = vertex;
            return;
        }

        // Not in cache, add to front and push others back
        for i in (1..self.cache_size).rev() {
            self.cache[i] = self.cache[i - 1];
        }
        self.cache[0] = vertex;
    }
}