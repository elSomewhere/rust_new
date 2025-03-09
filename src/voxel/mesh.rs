use std::collections::HashMap;
use glam::{Vec3, Vec2, IVec3};
use serde::{Serialize, Deserialize};

use crate::voxel::types::{VoxelData, VoxelInstance};
use crate::utils::VertexCacheOptimizer;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeshStrategy {
    Instanced,
    GreedyMesh,
    MarchingCubes,
    DualContouring,
    MeshShader,
}

impl MeshStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Instanced => "Instanced",
            Self::GreedyMesh => "Greedy Mesh",
            Self::MarchingCubes => "Marching Cubes",
            Self::DualContouring => "Dual Contouring",
            Self::MeshShader => "Mesh Shader",
        }
    }
}

// Vertex data for mesh rendering
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tex_coords: Vec2,
    pub color: [f32; 4],
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, tex_coords: Vec2, color: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            tex_coords,
            color,
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// Ensure Vertex can be used in GPU buffers
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

// Mesh data structure for various mesh generation strategies
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub instances: Vec<VoxelInstance>,
}

impl MeshData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            instances: Vec::new(),
        }
    }

    pub fn optimize(&mut self) {
        if self.indices.len() < 3 {
            return;
        }

        // Use vertex cache optimization
        let mut optimizer = VertexCacheOptimizer::new(32);
        optimizer.optimize(&mut self.indices, self.vertices.len());
    }

    pub fn add_face(&mut self, face_verts: &[Vertex; 4], face_indices: &[u32; 6]) {
        let base_index = self.vertices.len() as u32;

        // Add vertices
        self.vertices.extend_from_slice(face_verts);

        // Add indices with offset
        for index in face_indices {
            self.indices.push(base_index + index);
        }
    }

    pub fn add_triangle(&mut self, a: Vertex, b: Vertex, c: Vertex) {
        let base_index = self.vertices.len() as u32;

        self.vertices.push(a);
        self.vertices.push(b);
        self.vertices.push(c);

        self.indices.push(base_index);
        self.indices.push(base_index + 1);
        self.indices.push(base_index + 2);
    }

    pub fn add_instance(&mut self, instance: VoxelInstance) {
        self.instances.push(instance);
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.instances.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() && self.instances.is_empty()
    }
}

// Constants for cube vertices and faces
pub const CUBE_VERTICES: [Vec3; 8] = [
    Vec3::new(0.0, 0.0, 0.0), // 0: bottom-left-back
    Vec3::new(1.0, 0.0, 0.0), // 1: bottom-right-back
    Vec3::new(1.0, 0.0, 1.0), // 2: bottom-right-front
    Vec3::new(0.0, 0.0, 1.0), // 3: bottom-left-front
    Vec3::new(0.0, 1.0, 0.0), // 4: top-left-back
    Vec3::new(1.0, 1.0, 0.0), // 5: top-right-back
    Vec3::new(1.0, 1.0, 1.0), // 6: top-right-front
    Vec3::new(0.0, 1.0, 1.0), // 7: top-left-front
];

pub const CUBE_NORMALS: [Vec3; 6] = [
    Vec3::new(0.0, 0.0, -1.0), // Back
    Vec3::new(0.0, 0.0, 1.0),  // Front
    Vec3::new(-1.0, 0.0, 0.0), // Left
    Vec3::new(1.0, 0.0, 0.0),  // Right
    Vec3::new(0.0, -1.0, 0.0), // Bottom
    Vec3::new(0.0, 1.0, 0.0),  // Top
];

pub const CUBE_TEX_COORDS: [Vec2; 4] = [
    Vec2::new(0.0, 0.0), // Bottom-left
    Vec2::new(1.0, 0.0), // Bottom-right
    Vec2::new(1.0, 1.0), // Top-right
    Vec2::new(0.0, 1.0), // Top-left
];

// Face indices for each of the 6 faces (2 triangles per face)
pub const CUBE_FACE_INDICES: [[u32; 6]; 6] = [
    [0, 4, 5, 0, 5, 1], // Back face
    [3, 2, 6, 3, 6, 7], // Front face
    [0, 3, 7, 0, 7, 4], // Left face
    [1, 5, 6, 1, 6, 2], // Right face
    [0, 1, 2, 0, 2, 3], // Bottom face
    [4, 7, 6, 4, 6, 5], // Top face
];

// Face vertices for each of the 6 faces (4 vertices per face)
pub const CUBE_FACE_VERTICES: [[usize; 4]; 6] = [
    [0, 4, 5, 1], // Back face (-Z)
    [3, 2, 6, 7], // Front face (+Z)
    [0, 3, 7, 4], // Left face (-X)
    [1, 5, 6, 2], // Right face (+X)
    [0, 1, 2, 3], // Bottom face (-Y)
    [4, 5, 6, 7], // Top face (+Y)
];

// Directions for each face
pub const FACE_DIRECTIONS: [IVec3; 6] = [
    IVec3::new(0, 0, -1), // Back
    IVec3::new(0, 0, 1),  // Front
    IVec3::new(-1, 0, 0), // Left
    IVec3::new(1, 0, 0),  // Right
    IVec3::new(0, -1, 0), // Bottom
    IVec3::new(0, 1, 0),  // Top
];

// Meshing strategy implementations
pub struct MeshGenerator;

impl MeshGenerator {
    // Generate mesh using instanced approach
    pub fn generate_instanced_mesh(
        voxels: &[VoxelData],
        chunk_position: IVec3,
        chunk_size: i32,
    ) -> MeshData {
        let mut mesh = MeshData::new();

        // Create an instance for each visible voxel
        for x in 0..chunk_size {
            for y in 0..chunk_size {
                for z in 0..chunk_size {
                    let local_pos = IVec3::new(x, y, z);
                    let index = (x + y * chunk_size + z * chunk_size * chunk_size) as usize;

                    if index >= voxels.len() {
                        continue;
                    }

                    let voxel = voxels[index];

                    if voxel.is_air() {
                        continue;
                    }

                    // Check if voxel has any exposed face
                    let mut has_exposed_face = false;

                    for (_face_idx, dir) in FACE_DIRECTIONS.iter().enumerate() {
                        let adj_pos = local_pos + *dir;

                        if adj_pos.x < 0 || adj_pos.x >= chunk_size ||
                            adj_pos.y < 0 || adj_pos.y >= chunk_size ||
                            adj_pos.z < 0 || adj_pos.z >= chunk_size {
                            // Edge of chunk, assume visible
                            has_exposed_face = true;
                            break;
                        }

                        let adj_index = (adj_pos.x + adj_pos.y * chunk_size + adj_pos.z * chunk_size * chunk_size) as usize;

                        if adj_index >= voxels.len() {
                            has_exposed_face = true;
                            break;
                        }

                        let adj_voxel = voxels[adj_index];

                        if adj_voxel.is_air() || adj_voxel.is_transparent() {
                            has_exposed_face = true;
                            break;
                        }
                    }

                    if !has_exposed_face {
                        continue;
                    }

                    // Create instance for this voxel
                    let world_pos = Vec3::new(
                        (chunk_position.x * chunk_size + x) as f32,
                        (chunk_position.y * chunk_size + y) as f32,
                        (chunk_position.z * chunk_size + z) as f32,
                    );

                    let instance = VoxelInstance::new(
                        world_pos,
                        0, // No rotation
                        1.0, // Full scale
                        voxel.material_id as u16,
                        0, // No AO
                        0, // No custom data
                    );

                    mesh.add_instance(instance);
                }
            }
        }

        mesh
    }

    // Generate mesh using greedy meshing approach
    // Replace the generate_greedy_mesh function with this version
    // ADD this function to MeshGenerator in mesh.rs:
    fn get_face_vertices(pos: Vec3, size: Vec3, normal: Vec3) -> [Vec3; 4] {
        // This function creates correctly oriented face vertices based on the normal
        let u = if normal.x.abs() > 0.0 { Vec3::new(0.0, 0.0, 1.0) }
        else if normal.z.abs() > 0.0 { Vec3::new(1.0, 0.0, 0.0) }
        else { Vec3::new(1.0, 0.0, 0.0) };

        let v = if normal.y.abs() > 0.0 { Vec3::new(1.0, 0.0, 0.0) }
        else { Vec3::new(0.0, 1.0, 0.0) };

        let halfSize = size * 0.5;

        [
            pos - u * halfSize.x - v * halfSize.y, // Bottom left
            pos + u * halfSize.x - v * halfSize.y, // Bottom right
            pos + u * halfSize.x + v * halfSize.y, // Top right
            pos - u * halfSize.x + v * halfSize.y, // Top left
        ]
    }
    pub fn generate_greedy_mesh(
        voxels: &[VoxelData],
        chunk_position: IVec3,
        chunk_size: i32,
    ) -> MeshData {
        let mut mesh = MeshData::new();

        // For each axis direction (X, Y, Z)
        for axis in 0..3 {
            let u = (axis + 1) % 3;
            let v = (axis + 2) % 3;

            // For both positive and negative directions along each axis
            for direction in 0..2 {
                let axis_dir = match (axis, direction) {
                    (0, 0) => IVec3::new(-1, 0, 0), // -X
                    (0, 1) => IVec3::new(1, 0, 0),  // +X
                    (1, 0) => IVec3::new(0, -1, 0), // -Y
                    (1, 1) => IVec3::new(0, 1, 0),  // +Y
                    (2, 0) => IVec3::new(0, 0, -1), // -Z
                    (2, 1) => IVec3::new(0, 0, 1),  // +Z
                    _ => unreachable!(),
                };


                // Normal vector for this face direction
                let normal = Vec3::new(axis_dir.x as f32, axis_dir.y as f32, axis_dir.z as f32);

                // Starting slice depends on direction
                let slice_start = if direction == 0 { 0 } else { 0 };
                let slice_end = if direction == 0 { chunk_size - 1 } else { chunk_size };

                // For each slice along the axis
                for slice in slice_start..slice_end {
                    let mut mask = vec![false; (chunk_size * chunk_size) as usize];
                    let mut materials = vec![0u8; (chunk_size * chunk_size) as usize];

                    // Build 2D mask for this slice
                    for v_coord in 0..chunk_size {
                        for u_coord in 0..chunk_size {
                            // Current position
                            let mut pos = IVec3::ZERO;
                            match axis {
                                0 => { pos.x = if direction == 0 { slice } else { slice };
                                    pos.y = u_coord; pos.z = v_coord; },
                                1 => { pos.x = v_coord;
                                    pos.y = if direction == 0 { slice } else { slice };
                                    pos.z = u_coord; },
                                _ => { pos.x = u_coord; pos.y = v_coord;
                                    pos.z = if direction == 0 { slice } else { slice }; },
                            }

                            let index = (pos.x + pos.y * chunk_size + pos.z * chunk_size * chunk_size) as usize;
                            if index >= voxels.len() {
                                continue;
                            }

                            // Check for adjacency
                            let pos_adj = pos + axis_dir;

                            // Determine if adjacent voxel is air or solid
                            let current_voxel = voxels[index];
                            let adjacent_voxel = if pos_adj.x < 0 || pos_adj.x >= chunk_size ||
                                pos_adj.y < 0 || pos_adj.y >= chunk_size ||
                                pos_adj.z < 0 || pos_adj.z >= chunk_size {
                                // Always assume air at chunk boundaries - this will be fixed later but ensures edges render
                                VoxelData::air()
                            } else {
                                let adj_index = (pos_adj.x + pos_adj.y * chunk_size +
                                    pos_adj.z * chunk_size * chunk_size) as usize;
                                if adj_index >= voxels.len() {
                                    VoxelData::air()
                                } else {
                                    voxels[adj_index]
                                }
                            };

                            // A face is visible if:
                            // 1. Current voxel is not air
                            // 2. Adjacent voxel is air or transparent
                            let face_visible = !current_voxel.is_air() &&
                                (adjacent_voxel.is_air() || adjacent_voxel  .is_transparent());

                            let mask_index = (v_coord * chunk_size + u_coord) as usize;
                            mask[mask_index] = face_visible;
                            if face_visible {
                                materials[mask_index] = current_voxel.material_id;
                            }

                        }
                    }

                    // Generate quads using greedy meshing
                    let mut visited = vec![false; (chunk_size * chunk_size) as usize];

                    for v_coord in 0..chunk_size {
                        for u_coord in 0..chunk_size {
                            let mask_index = (v_coord * chunk_size + u_coord) as usize;

                            if !mask[mask_index] || visited[mask_index] {
                                continue;
                            }

                            let material = materials[mask_index];

                            // Find width of quad
                            let mut width = 1;
                            while u_coord + width < chunk_size {
                                let w_index = (v_coord * chunk_size + (u_coord + width)) as usize;

                                if !mask[w_index] || materials[w_index] != material || visited[w_index] {
                                    break;
                                }

                                width += 1;
                            }

                            // Find height of quad
                            let mut height = 1;
                            let mut can_extend = true;

                            while v_coord + height < chunk_size && can_extend {
                                for w in 0..width {
                                    let h_index = ((v_coord + height) * chunk_size + (u_coord + w)) as usize;

                                    if !mask[h_index] || materials[h_index] != material || visited[h_index] {
                                        can_extend = false;
                                        break;
                                    }
                                }

                                if can_extend {
                                    height += 1;
                                }
                            }

                            // Mark these cells as visited
                            for h in 0..height {
                                for w in 0..width {
                                    let visit_index = ((v_coord + h) * chunk_size + (u_coord + w)) as usize;
                                    visited[visit_index] = true;
                                }
                            }

                            // Generate quad vertices
                            let mut pos = IVec3::ZERO;
                            let mut pos2 = IVec3::ZERO;

                            match axis {
                                0 => { // X axis
                                    if direction == 0 {
                                        pos.x = slice;
                                        pos2.x = slice;
                                    } else {
                                        pos.x = slice + 1;
                                        pos2.x = slice + 1;
                                    }
                                    pos.y = u_coord;
                                    pos.z = v_coord;
                                    pos2.y = u_coord + width;
                                    pos2.z = v_coord + height;
                                },
                                1 => { // Y axis
                                    if direction == 0 {
                                        pos.y = slice;
                                        pos2.y = slice;
                                    } else {
                                        pos.y = slice + 1;
                                        pos2.y = slice + 1;
                                    }
                                    pos.x = v_coord;
                                    pos.z = u_coord;
                                    pos2.x = v_coord + height;
                                    pos2.z = u_coord + width;
                                },
                                _ => { // Z axis
                                    if direction == 0 {
                                        pos.z = slice;
                                        pos2.z = slice;
                                    } else {
                                        pos.z = slice + 1;
                                        pos2.z = slice + 1;
                                    }
                                    pos.x = u_coord;
                                    pos.y = v_coord;
                                    pos2.x = u_coord + width;
                                    pos2.y = v_coord + height;
                                },
                            }

                            // Generate color based on material
                            let color = match material {
                                0 => [1.0, 1.0, 1.0, 0.0],    // Air (transparent)
                                1 => [0.5, 0.5, 0.5, 1.0],    // Stone
                                2 => [0.6, 0.3, 0.1, 1.0],    // Dirt
                                3 => [0.3, 0.7, 0.2, 1.0],    // Grass
                                4 => [0.9, 0.8, 0.6, 1.0],    // Sand
                                5 => [0.2, 0.4, 0.8, 0.7],    // Water
                                6 => [0.5, 0.3, 0.1, 1.0],    // Wood
                                7 => [0.2, 0.5, 0.1, 0.9],    // Leaves
                                8 => [0.9, 0.8, 0.1, 1.0],    // Gold ore
                                9 => [0.6, 0.5, 0.5, 1.0],    // Iron ore
                                10 => [0.2, 0.2, 0.2, 1.0],   // Coal
                                _ => [1.0, 0.0, 1.0, 1.0],    // Unknown (magenta)
                            };

                            // With this more consistent approach:
                            let vertex_order = if direction == 0 {
                                [0, 3, 2, 1] // Consistent CCW for negative faces
                            } else {
                                [1, 2, 3, 0] // Consistent CCW for positive faces
                            };

                            // Create vertices for quad (using the correct winding order)
                            let face_corners = [
                                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),              // Bottom-left
                                Vec3::new(pos2.x as f32, pos.y as f32, pos.z as f32),             // Bottom-right
                                Vec3::new(pos2.x as f32, pos2.y as f32, pos2.z as f32),           // Top-right
                                Vec3::new(pos.x as f32, pos2.y as f32, pos2.z as f32),            // Top-left
                            ];
                            // Calculate correct world position by multiplying chunk_position by chunk_size
                            let chunk_world_pos = Vec3::new(
                                chunk_position.x as f32 * chunk_size as f32,
                                chunk_position.y as f32 * chunk_size as f32,
                                chunk_position.z as f32 * chunk_size as f32
                            );



                            let mut quad_verts = [
                                Vertex::new(face_corners[vertex_order[0]] + chunk_world_pos, normal, Vec2::new(0.0, 0.0), color),
                                Vertex::new(face_corners[vertex_order[1]] + chunk_world_pos, normal, Vec2::new(1.0, 0.0), color),
                                Vertex::new(face_corners[vertex_order[2]] + chunk_world_pos, normal, Vec2::new(1.0, 1.0), color),
                                Vertex::new(face_corners[vertex_order[3]] + chunk_world_pos, normal, Vec2::new(0.0, 1.0), color),
                            ];

                            // Offset vertices by chunk position
                            for vert in &mut quad_verts {
                                vert.position += Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                );
                            }

                            // Add quad to mesh using consistent triangle indices
                            let quad_indices = [0u32, 1, 2, 0, 2, 3];
                            mesh.add_face(&quad_verts, &quad_indices);
                        }
                    }
                }
            }
        }

        // Optimize the mesh
        mesh.optimize();

        mesh
    }

    // Generate mesh using marching cubes
    pub fn generate_marching_cubes_mesh(
        voxels: &[VoxelData],
        chunk_position: IVec3,
        chunk_size: i32,
    ) -> MeshData {
        let mut mesh = MeshData::new();

        // Marching cubes lookup tables
        let edge_table = Self::get_mc_edge_table();
        let tri_table = Self::get_mc_tri_table();

        // Process each cell (cube) in the grid
        for x in 0..chunk_size-1 {
            for y in 0..chunk_size-1 {
                for z in 0..chunk_size-1 {
                    // Get the eight corners of the current cube
                    let corner_positions = [
                        IVec3::new(x, y, z),
                        IVec3::new(x+1, y, z),
                        IVec3::new(x+1, y, z+1),
                        IVec3::new(x, y, z+1),
                        IVec3::new(x, y+1, z),
                        IVec3::new(x+1, y+1, z),
                        IVec3::new(x+1, y+1, z+1),
                        IVec3::new(x, y+1, z+1),
                    ];

                    // Get density values for each corner
                    let mut corner_values = [0.0f32; 8];
                    let mut corner_materials = [0u8; 8];

                    for i in 0..8 {
                        let pos = corner_positions[i];
                        let index = (pos.x + pos.y * chunk_size + pos.z * chunk_size * chunk_size) as usize;

                        if index < voxels.len() {
                            let voxel = voxels[index];
                            corner_values[i] = voxel.density as f32 / 255.0;
                            corner_materials[i] = voxel.material_id;
                        }
                    }

                    // Determine the index in the edge table
                    let mut cube_index = 0u8;
                    let iso_level = 0.5f32; // Threshold for the surface

                    for i in 0..8 {
                        if corner_values[i] >= iso_level {
                            cube_index |= 1 << i;
                        }
                    }

                    // If the cube is entirely inside or outside the surface, skip it
                    if edge_table[cube_index as usize] == 0 {
                        continue;
                    }

                    // Calculate intersection vertices where the surface crosses the cube edges
                    let mut intersection_verts = [Vec3::ZERO; 12];
                    let mut intersection_normals = [Vec3::ZERO; 12];

                    for i in 0..12 {
                        // Check if this edge is crossed
                        if (edge_table[cube_index as usize] & (1 << i)) != 0 {
                            // Get the two corner indices for this edge
                            let (v1, v2) = match i {
                                0 => (0, 1), 1 => (1, 2), 2 => (2, 3), 3 => (3, 0),
                                4 => (4, 5), 5 => (5, 6), 6 => (6, 7), 7 => (7, 4),
                                8 => (0, 4), 9 => (1, 5), 10 => (2, 6), 11 => (3, 7),
                                _ => unreachable!(),
                            };

                            // Interpolate position based on density values
                            let t = (iso_level - corner_values[v1]) / (corner_values[v2] - corner_values[v1]);

                            intersection_verts[i] = Vec3::new(
                                corner_positions[v1].x as f32 + t * (corner_positions[v2].x as f32 - corner_positions[v1].x as f32),
                                corner_positions[v1].y as f32 + t * (corner_positions[v2].y as f32 - corner_positions[v1].y as f32),
                                corner_positions[v1].z as f32 + t * (corner_positions[v2].z as f32 - corner_positions[v1].z as f32),
                            );

                            // Calculate normal by central differences
                            let epsilon = 0.01f32;
                            let central_pos = intersection_verts[i];

                            // Sample densities around this point
                            let sample_x_plus = Self::sample_density_at(
                                central_pos + Vec3::new(epsilon, 0.0, 0.0),
                                voxels,
                                chunk_size,
                            );

                            let sample_x_minus = Self::sample_density_at(
                                central_pos - Vec3::new(epsilon, 0.0, 0.0),
                                voxels,
                                chunk_size,
                            );

                            let sample_y_plus = Self::sample_density_at(
                                central_pos + Vec3::new(0.0, epsilon, 0.0),
                                voxels,
                                chunk_size,
                            );

                            let sample_y_minus = Self::sample_density_at(
                                central_pos - Vec3::new(0.0, epsilon, 0.0),
                                voxels,
                                chunk_size,
                            );

                            let sample_z_plus = Self::sample_density_at(
                                central_pos + Vec3::new(0.0, 0.0, epsilon),
                                voxels,
                                chunk_size,
                            );

                            let sample_z_minus = Self::sample_density_at(
                                central_pos - Vec3::new(0.0, 0.0, epsilon),
                                voxels,
                                chunk_size,
                            );

                            // Calculate gradient
                            let gradient = Vec3::new(
                                (sample_x_plus - sample_x_minus) / (2.0 * epsilon),
                                (sample_y_plus - sample_y_minus) / (2.0 * epsilon),
                                (sample_z_plus - sample_z_minus) / (2.0 * epsilon),
                            );

                            // Use gradient as normal
                            intersection_normals[i] = gradient.normalize();
                        }
                    }

                    // Determine dominant material
                    let mut material_counts = HashMap::new();
                    for &material in &corner_materials {
                        if material == 0 {
                            continue; // Skip air
                        }
                        *material_counts.entry(material).or_insert(0) += 1;
                    }

                    let dominant_material = material_counts
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(material, _)| material)
                        .unwrap_or(1); // Default to stone if no material is found

                    // Generate color based on material
                    let color = match dominant_material {
                        0 => [1.0, 1.0, 1.0, 0.0], // Air (transparent)
                        1 => [0.5, 0.5, 0.5, 1.0], // Stone
                        2 => [0.6, 0.3, 0.1, 1.0], // Dirt
                        3 => [0.3, 0.7, 0.2, 1.0], // Grass
                        4 => [0.9, 0.8, 0.6, 1.0], // Sand
                        5 => [0.2, 0.4, 0.8, 0.7], // Water
                        _ => [1.0, 0.0, 1.0, 1.0], // Unknown (magenta)
                    };

                    // Create triangles based on the marching cubes algorithm
                    let mut i = 0;
                    while tri_table[cube_index as usize][i] != -1 {
                        let v1 = tri_table[cube_index as usize][i] as usize;
                        let v2 = tri_table[cube_index as usize][i+1] as usize;
                        let v3 = tri_table[cube_index as usize][i+2] as usize;

                        // Add a triangle to the mesh
                        mesh.add_triangle(
                            Vertex::new(
                                intersection_verts[v1] + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                intersection_normals[v1],
                                Vec2::new(0.0, 0.0), // Placeholder UVs
                                color
                            ),
                            Vertex::new(
                                intersection_verts[v2] + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                intersection_normals[v2],
                                Vec2::new(1.0, 0.0), // Placeholder UVs
                                color
                            ),
                            Vertex::new(
                                intersection_verts[v3] + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                intersection_normals[v3],
                                Vec2::new(0.0, 1.0), // Placeholder UVs
                                color
                            ),
                        );

                        i += 3;
                    }
                }
            }
        }

        // Optimize the mesh
        mesh.optimize();

        mesh
    }

    // Simplified dual contouring implementation
    pub fn generate_dual_contouring_mesh(
        voxels: &[VoxelData],
        chunk_position: IVec3,
        chunk_size: i32,
    ) -> MeshData {
        // For simplicity, we'll use a simplified version of dual contouring
        // This implementation won't preserve sharp features but will demonstrate the basic concept
        let mut mesh = MeshData::new();

        // QEF solver is complex, so we'll use a simplified approach
        // Store vertex positions for each cell
        let mut cell_vertices = vec![None; ((chunk_size + 1) * (chunk_size + 1) * (chunk_size + 1)) as usize];
        let mut cell_materials = vec![0u8; ((chunk_size + 1) * (chunk_size + 1) * (chunk_size + 1)) as usize];

        // First, find cells that intersect the surface and place vertices
        for x in 0..chunk_size {
            for y in 0..chunk_size {
                for z in 0..chunk_size {
                    // Check the 8 corners of each cell
                    let corners = [
                        (x, y, z),
                        (x+1, y, z),
                        (x+1, y, z+1),
                        (x, y, z+1),
                        (x, y+1, z),
                        (x+1, y+1, z),
                        (x+1, y+1, z+1),
                        (x, y+1, z+1),
                    ];

                    // Get material for each corner
                    let mut corner_materials = [0u8; 8];
                    let mut all_same = true;
                    let mut first_material = 0u8;

                    for (i, &(cx, cy, cz)) in corners.iter().enumerate() {
                        let index = (cx + cy * chunk_size + cz * chunk_size * chunk_size) as usize;
                        if index < voxels.len() {
                            corner_materials[i] = voxels[index].material_id;

                            if i == 0 {
                                first_material = corner_materials[i];
                            } else if corner_materials[i] != first_material {
                                all_same = false;
                            }
                        }
                    }

                    // Skip if all corners have same material (no surface crossing)
                    if all_same {
                        continue;
                    }

                    // Find most common non-air material
                    let mut material_counts = HashMap::new();
                    for &material in &corner_materials {
                        if material == 0 {
                            continue; // Skip air
                        }
                        *material_counts.entry(material).or_insert(0) += 1;
                    }

                    let dominant_material = material_counts
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(material, _)| material)
                        .unwrap_or(1); // Default to stone if no material is found

                    // Check if we have air/solid transitions along edges
                    let mut has_transition = false;
                    let edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7),
                    ];

                    for &(a, b) in &edges {
                        let mat_a = corner_materials[a];
                        let mat_b = corner_materials[b];

                        if (mat_a == 0 && mat_b != 0) || (mat_a != 0 && mat_b == 0) {
                            has_transition = true;
                            break;
                        }
                    }

                    if !has_transition {
                        continue;
                    }

                    // Place a vertex at the cell center (simplified QEF)
                    let cell_center = Vec3::new(
                        x as f32 + 0.5,
                        y as f32 + 0.5,
                        z as f32 + 0.5,
                    );

                    let cell_index = (x + y * (chunk_size + 1) + z * (chunk_size + 1) * (chunk_size + 1)) as usize;
                    cell_vertices[cell_index] = Some(cell_center);
                    cell_materials[cell_index] = dominant_material;
                }
            }
        }

        // Generate quads connecting cell vertices
        for x in 0..chunk_size-1 {
            for y in 0..chunk_size-1 {
                for z in 0..chunk_size-1 {
                    // Vertices at the corners of a dual cell
                    let v000 = (x + y * (chunk_size + 1) + z * (chunk_size + 1) * (chunk_size + 1)) as usize;
                    let v100 = (x+1 + y * (chunk_size + 1) + z * (chunk_size + 1) * (chunk_size + 1)) as usize;
                    let v010 = (x + (y+1) * (chunk_size + 1) + z * (chunk_size + 1) * (chunk_size + 1)) as usize;
                    let v110 = (x+1 + (y+1) * (chunk_size + 1) + z * (chunk_size + 1) * (chunk_size + 1)) as usize;
                    let v001 = (x + y * (chunk_size + 1) + (z+1) * (chunk_size + 1) * (chunk_size + 1)) as usize;
                    let v101 = (x+1 + y * (chunk_size + 1) + (z+1) * (chunk_size + 1) * (chunk_size + 1)) as usize;
                    let v011 = (x + (y+1) * (chunk_size + 1) + (z+1) * (chunk_size + 1) * (chunk_size + 1)) as usize;
                    let v111 = (x+1 + (y+1) * (chunk_size + 1) + (z+1) * (chunk_size + 1) * (chunk_size + 1)) as usize;

                    let vertices = [
                        &cell_vertices[v000], &cell_vertices[v100], &cell_vertices[v010], &cell_vertices[v110],
                        &cell_vertices[v001], &cell_vertices[v101], &cell_vertices[v011], &cell_vertices[v111],
                    ];

                    let materials = [
                        cell_materials[v000], cell_materials[v100], cell_materials[v010], cell_materials[v110],
                        cell_materials[v001], cell_materials[v101], cell_materials[v011], cell_materials[v111],
                    ];

                    // Skip if any vertex is missing
                    if vertices.iter().any(|&v| v.is_none()) {
                        continue;
                    }

                    // Check for dominant material
                    let material_counts = materials.iter().fold(HashMap::new(), |mut map, &mat| {
                        if mat != 0 {
                            *map.entry(mat).or_insert(0) += 1;
                        }
                        map
                    });

                    if material_counts.is_empty() {
                        continue;
                    }

                    let dominant_material = material_counts
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(material, _)| material)
                        .unwrap_or(1);

                    // Generate color based on material
                    let color = match dominant_material {
                        0 => [1.0, 1.0, 1.0, 0.0], // Air (transparent)
                        1 => [0.5, 0.5, 0.5, 1.0], // Stone
                        2 => [0.6, 0.3, 0.1, 1.0], // Dirt
                        3 => [0.3, 0.7, 0.2, 1.0], // Grass
                        4 => [0.9, 0.8, 0.6, 1.0], // Sand
                        5 => [0.2, 0.4, 0.8, 0.7], // Water
                        _ => [1.0, 0.0, 1.0, 1.0], // Unknown (magenta)
                    };

                    // Generate the 6 faces of the dual cell if needed
                    // -X face
                    if Self::should_generate_face(materials[0], materials[2], materials[4], materials[6]) {
                        let normal = Vec3::new(-1.0, 0.0, 0.0);

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[0].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[4].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[6].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                        );

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[0].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[6].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[2].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 1.0),
                                color,
                            ),
                        );
                    }

                    // +X face
                    if Self::should_generate_face(materials[1], materials[3], materials[5], materials[7]) {
                        let normal = Vec3::new(1.0, 0.0, 0.0);

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[1].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[5].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[7].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                        );

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[1].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[7].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[3].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 1.0),
                                color,
                            ),
                        );
                    }

                    // Similar code for the other 4 faces (Y and Z directions)
                    // -Y face
                    if Self::should_generate_face(materials[0], materials[1], materials[4], materials[5]) {
                        let normal = Vec3::new(0.0, -1.0, 0.0);

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[0].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[1].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[5].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                        );

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[0].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[5].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[4].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 1.0),
                                color,
                            ),
                        );
                    }

                    // +Y face
                    if Self::should_generate_face(materials[2], materials[3], materials[6], materials[7]) {
                        let normal = Vec3::new(0.0, 1.0, 0.0);

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[2].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[3].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[7].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                        );

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[2].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[7].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[6].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 1.0),
                                color,
                            ),
                        );
                    }

                    // -Z face
                    if Self::should_generate_face(materials[0], materials[1], materials[2], materials[3]) {
                        let normal = Vec3::new(0.0, 0.0, -1.0);

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[0].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[1].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[3].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                        );

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[0].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[3].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[2].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 1.0),
                                color,
                            ),
                        );
                    }

                    // +Z face
                    if Self::should_generate_face(materials[4], materials[5], materials[6], materials[7]) {
                        let normal = Vec3::new(0.0, 0.0, 1.0);

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[4].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[5].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[7].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                        );

                        mesh.add_triangle(
                            Vertex::new(
                                vertices[4].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 0.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[7].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(1.0, 1.0),
                                color,
                            ),
                            Vertex::new(
                                vertices[6].unwrap() + Vec3::new(
                                    chunk_position.x as f32 * chunk_size as f32,
                                    chunk_position.y as f32 * chunk_size as f32,
                                    chunk_position.z as f32 * chunk_size as f32,
                                ),
                                normal,
                                Vec2::new(0.0, 1.0),
                                color,
                            ),
                        );
                    }
                }
            }
        }

        // Optimize the mesh
        mesh.optimize();

        mesh
    }

    // Helper function to check if a face should be generated for dual contouring
    fn should_generate_face(mat1: u8, mat2: u8, mat3: u8, mat4: u8) -> bool {
        // Generate face if we have a transition between air and solid
        let has_air = mat1 == 0 || mat2 == 0 || mat3 == 0 || mat4 == 0;
        let has_solid = mat1 != 0 || mat2 != 0 || mat3 != 0 || mat4 != 0;

        has_air && has_solid
    }

    // Helper function to sample density at a point
    fn sample_density_at(pos: Vec3, voxels: &[VoxelData], chunk_size: i32) -> f32 {
        // Convert to integer position
        let ix = pos.x.floor() as i32;
        let iy = pos.y.floor() as i32;
        let iz = pos.z.floor() as i32;

        // Make sure we're within bounds
        if ix < 0 || ix >= chunk_size ||
            iy < 0 || iy >= chunk_size ||
            iz < 0 || iz >= chunk_size {
            return 0.0;
        }

        let index = (ix + iy * chunk_size + iz * chunk_size * chunk_size) as usize;

        if index >= voxels.len() {
            return 0.0;
        }

        let voxel = voxels[index];
        voxel.density as f32 / 255.0
    }

    // Marching cubes edge table
    fn get_mc_edge_table() -> [u16; 256] {
        [
            0x0000, 0x0109, 0x0203, 0x030A, 0x0406, 0x050F, 0x0605, 0x070C,
            0x080C, 0x0905, 0x0A0F, 0x0B06, 0x0C0A, 0x0D03, 0x0E09, 0x0F00,
            0x0190, 0x0099, 0x0393, 0x029A, 0x0596, 0x049F, 0x0795, 0x069C,
            0x099C, 0x0895, 0x0B9F, 0x0A96, 0x0D9A, 0x0C93, 0x0F99, 0x0E90,
            0x0230, 0x0339, 0x0033, 0x013A, 0x0636, 0x073F, 0x0435, 0x053C,
            0x0A3C, 0x0B35, 0x083F, 0x0936, 0x0E3A, 0x0F33, 0x0C39, 0x0D30,
            0x03A0, 0x02A9, 0x01A3, 0x00AA, 0x07A6, 0x06AF, 0x05A5, 0x04AC,
            0x0BAC, 0x0AA5, 0x09AF, 0x08A6, 0x0FAA, 0x0EA3, 0x0DA9, 0x0CA0,
            0x0460, 0x0569, 0x0663, 0x076A, 0x0066, 0x016F, 0x0265, 0x036C,
            0x0C6C, 0x0D65, 0x0E6F, 0x0F66, 0x086A, 0x0963, 0x0A69, 0x0B60,
            0x05F0, 0x04F9, 0x07F3, 0x06FA, 0x01F6, 0x00FF, 0x03F5, 0x02FC,
            0x0DFC, 0x0CF5, 0x0FFF, 0x0EF6, 0x09FA, 0x08F3, 0x0BF9, 0x0AF0,
            0x0650, 0x0759, 0x0453, 0x055A, 0x0256, 0x035F, 0x0055, 0x015C,
            0x0E5C, 0x0F55, 0x0C5F, 0x0D56, 0x0A5A, 0x0B53, 0x0859, 0x0950,
            0x07C0, 0x06C9, 0x05C3, 0x04CA, 0x03C6, 0x02CF, 0x01C5, 0x00CC,
            0x0FCC, 0x0EC5, 0x0DCF, 0x0CC6, 0x0BCA, 0x0AC3, 0x09C9, 0x08C0,
            0x08C0, 0x09C9, 0x0AC3, 0x0BCA, 0x0CC6, 0x0DCF, 0x0EC5, 0x0FCC,
            0x00CC, 0x01C5, 0x02CF, 0x03C6, 0x04CA, 0x05C3, 0x06C9, 0x07C0,
            0x0950, 0x0859, 0x0B53, 0x0A5A, 0x0D56, 0x0C5F, 0x0F55, 0x0E5C,
            0x015C, 0x0055, 0x035F, 0x0256, 0x055A, 0x0453, 0x0759, 0x0650,
            0x0AF0, 0x0BF9, 0x08F3, 0x09FA, 0x0EF6, 0x0FFF, 0x0CF5, 0x0DFC,
            0x02FC, 0x03F5, 0x00FF, 0x01F6, 0x06FA, 0x07F3, 0x04F9, 0x05F0,
            0x0B60, 0x0A69, 0x0963, 0x086A, 0x0F66, 0x0E6F, 0x0D65, 0x0C6C,
            0x036C, 0x0265, 0x016F, 0x0066, 0x076A, 0x0663, 0x0569, 0x0460,
            0x0CA0, 0x0DA9, 0x0EA3, 0x0FAA, 0x08A6, 0x09AF, 0x0AA5, 0x0BAC,
            0x04AC, 0x05A5, 0x06AF, 0x07A6, 0x00AA, 0x01A3, 0x02A9, 0x03A0,
            0x0D30, 0x0C39, 0x0F33, 0x0E3A, 0x0936, 0x083F, 0x0B35, 0x0A3C,
            0x053C, 0x0435, 0x073F, 0x0636, 0x013A, 0x0033, 0x0339, 0x0230,
            0x0E90, 0x0F99, 0x0C93, 0x0D9A, 0x0A96, 0x0B9F, 0x0895, 0x099C,
            0x069C, 0x0795, 0x049F, 0x0596, 0x029A, 0x0393, 0x0099, 0x0190,
            0x0F00, 0x0E09, 0x0D03, 0x0C0A, 0x0B06, 0x0A0F, 0x0905, 0x080C,
            0x070C, 0x0605, 0x050F, 0x0406, 0x030A, 0x0203, 0x0109, 0x0000
        ]
    }

    // Marching cubes triangle table
    fn get_mc_tri_table() -> [[i8; 16]; 256] {
        // This is a simplified version of the full table
        // Full table would be too large to include here
        let mut table = [[0i8; 16]; 256];

        // Terminator
        for i in 0..256 {
            table[i][15] = -1;
        }

        // Define some common configurations
        // Case 1: One corner inside (one triangle)
        table[1] = [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[2] = [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[4] = [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[8] = [2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[16] = [3, 0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[32] = [4, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[64] = [5, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[128] = [6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

        // Case 3: Two adjacent corners inside (one quad = two triangles)
        table[3] = [0, 1, 9, 9, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[6] = [1, 2, 10, 10, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[12] = [2, 3, 11, 11, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[24] = [3, 0, 8, 8, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[48] = [4, 5, 6, 6, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[96] = [5, 6, 11, 11, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        table[192] = [6, 7, 4, 4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

        // Case 7: Three corners inside (one pentagon = three triangles)
        table[7] = [1, 2, 10, 10, 9, 1, 9, 10, 8, -1, -1, -1, -1, -1, -1, -1];
        table[14] = [2, 3, 11, 11, 10, 2, 10, 11, 9, -1, -1, -1, -1, -1, -1, -1];
        table[28] = [3, 0, 8, 8, 11, 3, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1];
        table[56] = [0, 1, 9, 9, 8, 0, 8, 9, 11, -1, -1, -1, -1, -1, -1, -1];

        // Case 15: Four corners inside (one hexagon = four triangles)
        table[15] = [0, 1, 5, 5, 4, 0, 4, 5, 8, 8, 5, 10, 10, 8, 11, -1];

        // Many more cases would be defined here for a full implementation

        table
    }
}