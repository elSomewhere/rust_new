use std::sync::Arc;
use log::{info, warn};
use glam::{Vec3, IVec3};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::InputState;
use crate::UserAction;
use crate::voxel::{World, types::VoxelData};
use crate::rendering::{Camera, CameraController, RenderingSystem, CameraUniform, RenderContext};
use crate::worker::WorkerSystem;
use crate::physics::PhysicsSystem;

pub struct Engine {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    adapter: wgpu::Adapter,
    window: &'static Window,
    window_id: winit::window::WindowId,

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    world: World,
    rendering_system: RenderingSystem,
    worker_system: WorkerSystem,
    physics_system: PhysicsSystem,

    current_mesh_strategy: crate::voxel::mesh::MeshStrategy,
    frame_counter: u64,
}

impl Engine {
    pub async fn new(window: &'static Window) -> Self {
        let window_id = window.id();
        let size = window.inner_size();
        info!("Creating Engine with window size: {:?}", size);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        info!("WGPU Instance created");

        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");
        info!("Surface created");

        // Configure surface immediately to ensure it's valid
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");
        info!("Adapter requested successfully: {:?}", adapter.get_info());

        let surface_caps = surface.get_capabilities(&adapter);
        info!("Surface capabilities: {:?}", surface_caps);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");
        info!("Device and queue created");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        info!("Selected surface format: {:?}", surface_format);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        info!("Surface configured with {:?}", config);

        let camera = Camera::new(
            Vec3::new(10.0, 70.0, 10.0),  // Lower starting position (just above sea level at 60)
            -45.0_f32.to_radians(),
            -10.0_f32.to_radians(),       // Look more horizontally to see terrain better
            config.width as f32 / config.height as f32,
        );

        let camera_controller = CameraController::new(5.0, 0.5);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            },
        );

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let render_context = RenderContext::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            surface_format,
            camera_bind_group_layout,
        );

        let world = World::new();
        let rendering_system = RenderingSystem::new(render_context);
        let worker_system = WorkerSystem::new(Arc::clone(&device), Arc::clone(&queue));
        let physics_system = PhysicsSystem::new();
        let current_mesh_strategy = crate::voxel::mesh::MeshStrategy::Instanced;

        Self {
            surface,
            device,
            queue,
            config,
            adapter,
            window,
            window_id,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            world,
            rendering_system,
            worker_system,
            physics_system,
            current_mesh_strategy,
            frame_counter: 0,
        }
    }

    pub fn window_id(&self) -> winit::window::WindowId {
        self.window_id
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = new_size.width.max(1);
        self.config.height = new_size.height.max(1);
        self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        self.rendering_system.resize(new_size.width, new_size.height);
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
        self.frame_counter += 1;

        if let Some(action) = &input.action_triggered {
            match action {
                UserAction::ChangeStrategy(strategy) => {
                    self.current_mesh_strategy = *strategy;
                    self.world.set_mesh_strategy(self.current_mesh_strategy);
                    info!("Changed mesh strategy to {:?}", strategy);
                }
                UserAction::DestroyVoxel => {
                    let hit = self.physics_system.ray_cast(
                        &self.world,
                        self.camera.position,
                        self.camera.get_view_direction(),
                        10.0,
                    );
                    if let Some((pos, _)) = hit {
                        self.world.modify_voxel(pos, VoxelData::air());
                        info!("Removed voxel at {:?}", pos);
                    }
                }
                UserAction::CreateExplosion => {
                    let pos = self.camera.position + self.camera.get_view_direction() * 10.0;
                    let pos = IVec3::new(pos.x as i32, pos.y as i32, pos.z as i32);
                    self.world.create_explosion(pos, 5);
                    info!("Created explosion at {:?}", pos);
                }
            }
        }

        self.camera_controller.update_camera(&mut self.camera, input, dt);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Calculate the camera chunk position using floor division (not truncation)
        // This ensures proper chunk alignment
        let camera_chunk_pos = IVec3::new(
            (self.camera.position.x / 16.0).floor() as i32,
            (self.camera.position.y / 16.0).floor() as i32,
            (self.camera.position.z / 16.0).floor() as i32,
        );

        // Calculate a view-aligned offset but with reduced influence
        // to avoid creating gaps in the terrain
        let view_dir = self.camera.get_view_direction();
        let view_offset = IVec3::new(
            (view_dir.x * 1.0).round() as i32, // Reduced from 1.5 to 1.0
            (view_dir.y * 0.5).round() as i32, // Reduced vertical bias
            (view_dir.z * 1.0).round() as i32  // Reduced from 1.5 to 1.0
        );

        // Offset the chunk center slightly in the view direction
        let biased_chunk_pos = camera_chunk_pos + view_offset;

        // Update with an increased view distance
        let view_distance = 6; // Increased from default 4 in the World struct
        self.world.update(biased_chunk_pos, &self.camera, &mut self.worker_system);
        self.physics_system.update(&mut self.world, dt);
        self.worker_system.update();
    }

    pub fn render(&mut self) {
        let current_size = self.window.inner_size();
        if self.config.width != current_size.width || self.config.height != current_size.height {
            self.config.width = current_size.width.max(1);
            self.config.height = current_size.height.max(1);
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
            self.rendering_system.resize(self.config.width, self.config.height);
            info!("Resizing surface to {:?}", (self.config.width, self.config.height));
        }

        self.surface.configure(&self.device, &self.config);

        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                info!("Surface lost or outdated, reconfiguring");
                self.surface.configure(&self.device, &self.config);
                self.surface
                    .get_current_texture()
                    .expect("Failed to recover surface")
            }
            Err(wgpu::SurfaceError::Timeout) => {
                warn!("Surface timeout, skipping frame");
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                panic!("Out of memory during surface texture acquisition");
            }
            Err(wgpu::SurfaceError::Other) => {
                warn!("Unknown surface error, skipping frame");
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        self.rendering_system.render(
            &mut encoder,
            &view,
            &self.world,
            &self.camera,
            &self.camera_bind_group,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}