use std::sync::Arc;
use log::{info, warn};
use glam::{Vec3, IVec3};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::InputState;
use crate::UserAction;
use crate::voxel::{
    World,
    types::VoxelData
};
use crate::rendering::{
    Camera,
    CameraController,
    RenderingSystem,
    CameraUniform,
    RenderContext
};
use crate::worker::WorkerSystem;
use crate::physics::PhysicsSystem;

pub struct Engine {
    // WGPU core
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,

    // Window handle and info
    window_id: winit::window::WindowId,

    // Camera
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // Voxel world
    world: World,

    // Systems
    rendering_system: RenderingSystem,
    worker_system: WorkerSystem,
    physics_system: PhysicsSystem,

    // Runtime state
    current_mesh_strategy: crate::voxel::mesh::MeshStrategy,
    frame_counter: u64,
}

impl Engine {
    pub async fn new(window: &Window, surface: wgpu::Surface<'static>) -> Self {
        // Capture the window ID
        let window_id = window.id();

        // Initialize wgpu
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(), // Add features as needed
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ).await.expect("Failed to create device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Create camera
        let camera = Camera::new(
            Vec3::new(10.0, 100.0, 10.0),
            -45.0_f32.to_radians(),
            -20.0_f32.to_radians(),
            config.width as f32 / config.height as f32,
        );

        let camera_controller = CameraController::new(5.0, 0.5);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                ],
                label: Some("camera_bind_group_layout"),
            }
        );

        let camera_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }
                ],
                label: Some("camera_bind_group"),
            }
        );

        // Create rendering context
        let render_context = RenderContext::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            surface_format,
            camera_bind_group_layout,
        );

        // Initialize world and systems
        let world = World::new();

        let rendering_system = RenderingSystem::new(render_context);

        let worker_system = WorkerSystem::new(
            Arc::clone(&device),
            Arc::clone(&queue),
        );

        let physics_system = PhysicsSystem::new();

        let current_mesh_strategy = crate::voxel::mesh::MeshStrategy::Instanced;

        // Return the engine
        Self {
            surface,
            device,
            queue,
            config,
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

    // Add methods to modify window properties
    pub fn set_title(&self, window: &Window, title: &str) {
        window.set_title(title);
    }

    // Add getter for window ID
    pub fn window_id(&self) -> winit::window::WindowId {
        self.window_id
    }

    // Add getter for window size
    pub fn inner_size(&self, window: &Window) -> winit::dpi::PhysicalSize<u32> {
        window.inner_size()
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;

            // Update the depth texture in the rendering system
            self.rendering_system.resize(new_size.width, new_size.height);
        }
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
        // Update frame counter
        self.frame_counter += 1;

        // Handle user actions
        if let Some(action) = &input.action_triggered {
            match action {
                UserAction::ChangeStrategy(strategy) => {
                    self.current_mesh_strategy = *strategy;
                    self.world.set_mesh_strategy(*strategy);
                    info!("Changed mesh strategy to {:?}", strategy);
                },
                UserAction::DestroyVoxel => {
                    let hit = self.physics_system.ray_cast(&self.world,
                                                           self.camera.position, self.camera.get_view_direction(), 10.0);

                    if let Some((pos, _)) = hit {
                        self.world.modify_voxel(pos, VoxelData::air());
                        info!("Removed voxel at {:?}", pos);
                    }
                },
                UserAction::CreateExplosion => {
                    let pos = self.camera.position + self.camera.get_view_direction() * 10.0;
                    let pos = IVec3::new(pos.x as i32, pos.y as i32, pos.z as i32);
                    self.world.create_explosion(pos, 5);
                    info!("Created explosion at {:?}", pos);
                }
            }
        }

        // Update camera
        self.camera_controller.update_camera(&mut self.camera, input, dt);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update world based on camera position
        let camera_chunk_pos = IVec3::new(
            (self.camera.position.x / 16.0).floor() as i32,
            (self.camera.position.y / 16.0).floor() as i32,
            (self.camera.position.z / 16.0).floor() as i32,
        );
        self.world.update(camera_chunk_pos, &self.camera, &mut self.worker_system);

        // Update physics
        self.physics_system.update(&mut self.world, dt);

        // Update worker system
        self.worker_system.update();
    }

    pub fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(e) => {
                warn!("Failed to get current texture: {:?}", e);
                return;
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Render the world
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