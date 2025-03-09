use wgpu::util::DeviceExt;

use crate::voxel::mesh::{Vertex, CUBE_VERTICES, CUBE_NORMALS, CUBE_TEX_COORDS, CUBE_FACE_VERTICES, CUBE_FACE_INDICES};
use glam::{Vec3, Vec2};

pub struct CubeModel {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

// Create a cube model for instanced rendering
pub fn create_cube_model(device: &wgpu::Device) -> CubeModel {
    // Create vertices for the cube
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Generate vertices and indices for each face
    for face in 0..6 {
        let normal = CUBE_NORMALS[face];

        // Get the four vertices for this face
        let face_vertices = CUBE_FACE_VERTICES[face];
        let mut face_verts = [Vertex::default(); 4];

        for i in 0..4 {
            face_verts[i] = Vertex {
                position: CUBE_VERTICES[face_vertices[i]],
                normal,
                tex_coords: CUBE_TEX_COORDS[i],
                color: [1.0, 1.0, 1.0, 1.0],
            };
        }

        // Add vertices
        let base_index = vertices.len() as u32;
        vertices.extend_from_slice(&face_verts);

        // Add indices
        for idx in &CUBE_FACE_INDICES[face] {
            indices.push(base_index + idx);
        }
    }

    // Create vertex buffer
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cube Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Create index buffer
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cube Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let index_count = indices.len() as u32;

    CubeModel {
        vertex_buffer,
        index_buffer,
        index_count,
    }
}

// Create a depth texture for depth testing
pub fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let desc = wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };

    let texture = device.create_texture(&desc);

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}