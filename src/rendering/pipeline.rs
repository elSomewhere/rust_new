use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::voxel::{World, mesh::{MeshStrategy, Vertex, MeshData}};
use crate::voxel::types::{VoxelInstance, Material};
use crate::voxel::chunk::ChunkState;
use crate::rendering::camera::Camera;
use crate::rendering::resources::{create_cube_model, create_depth_texture, CubeModel};
use crate::rendering::shaders::{get_instance_shader, get_mesh_shader};

const MAX_INSTANCES: u32 = 100000;

// Render Context holds common rendering resources
pub struct RenderContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface_format: wgpu::TextureFormat,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
}

impl RenderContext {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: wgpu::BindGroupLayout,
    ) -> Self {
        Self {
            device,
            queue,
            surface_format,
            camera_bind_group_layout,
        }
    }
}

// Main rendering system
pub struct RenderingSystem {
    // Instanced rendering resources
    instance_pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
    cube_model: CubeModel,

    // Mesh rendering resources
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_buffers: MeshBuffers,

    // Common rendering resources
    depth_texture: wgpu::TextureView,
    materials: Vec<Material>,
    material_buffer: wgpu::Buffer,
    material_bind_group: wgpu::BindGroup,

    // Context
    context: RenderContext,
}

// Buffers for mesh rendering
struct MeshBuffers {
    greedy_vbuffer: Option<wgpu::Buffer>,
    greedy_ibuffer: Option<wgpu::Buffer>,
    greedy_index_count: u32,

    marching_vbuffer: Option<wgpu::Buffer>,
    marching_ibuffer: Option<wgpu::Buffer>,
    marching_index_count: u32,

    dual_vbuffer: Option<wgpu::Buffer>,
    dual_ibuffer: Option<wgpu::Buffer>,
    dual_index_count: u32,
}

impl MeshBuffers {
    fn new() -> Self {
        Self {
            greedy_vbuffer: None,
            greedy_ibuffer: None,
            greedy_index_count: 0,

            marching_vbuffer: None,
            marching_ibuffer: None,
            marching_index_count: 0,

            dual_vbuffer: None,
            dual_ibuffer: None,
            dual_index_count: 0,
        }
    }
}

// Implementation of VoxelInstance vertex buffer layout
impl VoxelInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Uint8,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Uint16,
                },
                wgpu::VertexAttribute {
                    offset: 22,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Uint8,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Uint16,
                },
            ],
        }
    }
}

impl RenderingSystem {
    pub fn new(context: RenderContext) -> Self {
        let device = &context.device;

        // Create cube model for instanced rendering
        let cube_model = create_cube_model(device);

        // Create instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<VoxelInstance>() as wgpu::BufferAddress * MAX_INSTANCES as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        // Create materials buffer
        let materials = Material::default_materials();
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Buffer"),
            contents: unsafe {
                std::slice::from_raw_parts(
                    materials.as_ptr() as *const u8,
                    std::mem::size_of::<Material>() * materials.len(),
                )
            },
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create material bind group layout
        let material_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create material bind group
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Bind Group"),
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        });

        // Load shaders
        let instance_shader = get_instance_shader(device);
        let mesh_shader = get_mesh_shader(device);

        // Create instance render pipeline layout
        let instance_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Instance Pipeline Layout"),
            bind_group_layouts: &[
                &context.camera_bind_group_layout,
                &material_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Create mesh render pipeline layout
        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[
                &context.camera_bind_group_layout,
                &material_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Create instanced render pipeline
        let instance_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instance Pipeline"),
            layout: Some(&instance_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &instance_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex::desc(),         // Vertex buffer
                    VoxelInstance::desc(),  // Instance buffer
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,//Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // Changed from LessEqual to Less
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &instance_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Create mesh render pipeline
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex::desc(),  // Vertex buffer
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // Changed from LessEqual to Less
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            }, // Fixed: removed extra closing parenthesis
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Create depth texture for depth testing
        let depth_texture = create_depth_texture(device, 1280, 720);

        Self {
            instance_pipeline,
            instance_buffer,
            instance_count: 0,
            cube_model,

            mesh_pipeline,
            mesh_buffers: MeshBuffers::new(),

            depth_texture,
            materials,
            material_buffer,
            material_bind_group,

            context,
        }
    }

    // Update the depth texture when window is resized
    pub fn resize(&mut self, width: u32, height: u32) {
        self.depth_texture = create_depth_texture(&self.context.device, width, height);
    }

    // Main render function
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        world: &World,
        camera: &Camera,
        camera_bind_group: &wgpu::BindGroup,
    ) {
        // Get active chunks
        let chunks = world.get_chunks();
        let chunks_guard = chunks.read().unwrap();
        let active_chunks = chunks_guard.get_active_chunks();

        // Collect instances from all chunks
        // FIX: Add explicit type annotation to instances vector
        let mut instances: Vec<VoxelInstance> = Vec::new();
        let mut greedy_mesh: Option<&MeshData> = None;
        let mut marching_mesh: Option<&MeshData> = None;
        let mut dual_mesh: Option<&MeshData> = None;

        for (_, chunk) in &active_chunks {
            if chunk.state != ChunkState::Ready {
                continue;
            }

            // Based on mesh strategy, collect mesh data
            match chunk.active_mesh_strategy {
                MeshStrategy::Instanced => {
                    if let Some(mesh) = chunk.mesh_data.get(&MeshStrategy::Instanced) {
                        instances.extend(&mesh.instances);
                    }
                },
                MeshStrategy::GreedyMesh => {
                    if let Some(mesh) = chunk.mesh_data.get(&MeshStrategy::GreedyMesh) {
                        if !mesh.is_empty() {
                            greedy_mesh = Some(mesh);
                        }
                    }
                },
                MeshStrategy::MarchingCubes => {
                    if let Some(mesh) = chunk.mesh_data.get(&MeshStrategy::MarchingCubes) {
                        if !mesh.is_empty() {
                            marching_mesh = Some(mesh);
                        }
                    }
                },
                MeshStrategy::DualContouring => {
                    if let Some(mesh) = chunk.mesh_data.get(&MeshStrategy::DualContouring) {
                        if !mesh.is_empty() {
                            dual_mesh = Some(mesh);
                        }
                    }
                },
                _ => {},
            }
        }

        // Limit instance count to MAX_INSTANCES
        if instances.len() > MAX_INSTANCES as usize {
            instances.truncate(MAX_INSTANCES as usize);
        }

        // Update instance buffer
        if !instances.is_empty() {
            self.context.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instances),
            );
            self.instance_count = instances.len() as u32;
        } else {
            self.instance_count = 0;
        }

        // Update mesh buffers if needed
        if let Some(mesh) = greedy_mesh {
            self.update_mesh_buffers(MeshStrategy::GreedyMesh, mesh);
        }

        if let Some(mesh) = marching_mesh {
            self.update_mesh_buffers(MeshStrategy::MarchingCubes, mesh);
        }

        if let Some(mesh) = dual_mesh {
            self.update_mesh_buffers(MeshStrategy::DualContouring, mesh);
        }

        // Begin render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.2,
                        g: 0.3,
                        b: 0.4,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Bind common resources
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.material_bind_group, &[]);

        // Render instanced cubes
        if self.instance_count > 0 {
            render_pass.set_pipeline(&self.instance_pipeline);
            render_pass.set_vertex_buffer(0, self.cube_model.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(
                self.cube_model.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..self.cube_model.index_count, 0, 0..self.instance_count);
        }

        // Render greedy mesh
        if let Some(vbuffer) = &self.mesh_buffers.greedy_vbuffer {
            if let Some(ibuffer) = &self.mesh_buffers.greedy_ibuffer {
                render_pass.set_pipeline(&self.mesh_pipeline);
                render_pass.set_vertex_buffer(0, vbuffer.slice(..));
                render_pass.set_index_buffer(ibuffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.mesh_buffers.greedy_index_count, 0, 0..1);
            }
        }

        // Render marching cubes mesh
        if let Some(vbuffer) = &self.mesh_buffers.marching_vbuffer {
            if let Some(ibuffer) = &self.mesh_buffers.marching_ibuffer {
                render_pass.set_pipeline(&self.mesh_pipeline);
                render_pass.set_vertex_buffer(0, vbuffer.slice(..));
                render_pass.set_index_buffer(ibuffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.mesh_buffers.marching_index_count, 0, 0..1);
            }
        }

        // Render dual contouring mesh
        if let Some(vbuffer) = &self.mesh_buffers.dual_vbuffer {
            if let Some(ibuffer) = &self.mesh_buffers.dual_ibuffer {
                render_pass.set_pipeline(&self.mesh_pipeline);
                render_pass.set_vertex_buffer(0, vbuffer.slice(..));
                render_pass.set_index_buffer(ibuffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.mesh_buffers.dual_index_count, 0, 0..1);
            }
        }
    }

    // Update mesh buffers for a specific strategy
    fn update_mesh_buffers(&mut self, strategy: MeshStrategy, mesh: &MeshData) {
        let device = &self.context.device;

        // Skip if mesh is empty
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return;
        }

        // Create vertex and index buffers
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Vertex Buffer", strategy)),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", strategy)),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Store buffers based on strategy
        match strategy {
            MeshStrategy::GreedyMesh => {
                self.mesh_buffers.greedy_vbuffer = Some(vertex_buffer);
                self.mesh_buffers.greedy_ibuffer = Some(index_buffer);
                self.mesh_buffers.greedy_index_count = mesh.indices.len() as u32;
            },
            MeshStrategy::MarchingCubes => {
                self.mesh_buffers.marching_vbuffer = Some(vertex_buffer);
                self.mesh_buffers.marching_ibuffer = Some(index_buffer);
                self.mesh_buffers.marching_index_count = mesh.indices.len() as u32;
            },
            MeshStrategy::DualContouring => {
                self.mesh_buffers.dual_vbuffer = Some(vertex_buffer);
                self.mesh_buffers.dual_ibuffer = Some(index_buffer);
                self.mesh_buffers.dual_index_count = mesh.indices.len() as u32;
            },
            _ => {},
        }
    }
}