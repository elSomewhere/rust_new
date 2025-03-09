use std::time::{Duration, Instant};
use log::info;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
    keyboard::{KeyCode, PhysicalKey},
};
use wgpu;

mod engine;
mod voxel;
mod rendering;
mod physics;
mod worker;
mod utils;

use engine::Engine;
use voxel::mesh::MeshStrategy;

#[derive(Debug)]
pub enum UserAction {
    ChangeStrategy(MeshStrategy),
    DestroyVoxel,
    CreateExplosion,
}

pub struct InputState {
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_run_pressed: bool,

    mouse_dx: f32,
    mouse_dy: f32,

    action_triggered: Option<UserAction>,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_run_pressed: false,

            mouse_dx: 0.0,
            mouse_dy: 0.0,

            action_triggered: None,
        }
    }
}

impl InputState {
    fn reset_mouse_delta(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
    }

    fn reset_actions(&mut self) {
        self.action_triggered = None;
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // Create window - but we'll move it into the event loop's closure below
    let window = WindowBuilder::new()
        .with_title("Voxel Engine")
        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .expect("Failed to create window");

    // Run the event loop - move the window into the closure
    event_loop.run(move |event, elwt| {
        // Initialize the window setup once
        static mut ENGINE: Option<Engine> = None;
        static mut INPUT_STATE: Option<InputState> = None;
        static mut LAST_FRAME_TIME: Option<Instant> = None;
        static mut FPS_COUNTER: Option<FpsCounter> = None;

        // One-time setup when the event loop starts
        if unsafe { ENGINE.is_none() } {
            // Grab the cursor
            window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
                .expect("Failed to grab cursor");
            window.set_cursor_visible(false);

            // Create WGPU instance
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            // Create surface
            let surface = instance.create_surface(&window).expect("Failed to create surface");

            // Initialize engine
            unsafe {
                ENGINE = Some(pollster::block_on(Engine::new(&window, surface)));
                INPUT_STATE = Some(InputState::default());
                LAST_FRAME_TIME = Some(Instant::now());
                FPS_COUNTER = Some(FpsCounter::new());
            }
        }

        // Get references to our static variables
        let engine = unsafe { ENGINE.as_mut().unwrap() };
        let input_state = unsafe { INPUT_STATE.as_mut().unwrap() };
        let last_frame_time = unsafe { LAST_FRAME_TIME.as_mut().unwrap() };
        let fps_counter = unsafe { FPS_COUNTER.as_mut().unwrap() };

        match event {
            Event::WindowEvent { ref event, window_id }
            if window_id == engine.window_id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    },
                    WindowEvent::Resized(physical_size) => {
                        engine.resize(*physical_size);
                    },
                    WindowEvent::ScaleFactorChanged { .. } => {
                        engine.resize(window.inner_size());
                    },
                    WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            state,
                            physical_key: PhysicalKey::Code(keycode),
                            ..
                        },
                        ..
                    } => {
                        let pressed = state == &ElementState::Pressed;
                        match keycode {
                            KeyCode::KeyW | KeyCode::ArrowUp => {
                                input_state.is_forward_pressed = pressed;
                            },
                            KeyCode::KeyS | KeyCode::ArrowDown => {
                                input_state.is_backward_pressed = pressed;
                            },
                            KeyCode::KeyA | KeyCode::ArrowLeft => {
                                input_state.is_left_pressed = pressed;
                            },
                            KeyCode::KeyD | KeyCode::ArrowRight => {
                                input_state.is_right_pressed = pressed;
                            },
                            KeyCode::Space => {
                                if pressed {
                                    input_state.action_triggered = Some(UserAction::CreateExplosion);
                                } else {
                                    input_state.is_up_pressed = false;
                                }
                            },
                            KeyCode::ShiftLeft => {
                                input_state.is_run_pressed = pressed;
                            },
                            KeyCode::ControlLeft => {
                                input_state.is_down_pressed = pressed;
                            },
                            KeyCode::Digit1 => {
                                if pressed {
                                    input_state.action_triggered = Some(UserAction::ChangeStrategy(MeshStrategy::Instanced));
                                }
                            },
                            KeyCode::Digit2 => {
                                if pressed {
                                    input_state.action_triggered = Some(UserAction::ChangeStrategy(MeshStrategy::GreedyMesh));
                                }
                            },
                            KeyCode::Digit3 => {
                                if pressed {
                                    input_state.action_triggered = Some(UserAction::ChangeStrategy(MeshStrategy::MarchingCubes));
                                }
                            },
                            KeyCode::Digit4 => {
                                if pressed {
                                    input_state.action_triggered = Some(UserAction::ChangeStrategy(MeshStrategy::DualContouring));
                                }
                            },
                            KeyCode::Escape => {
                                elwt.exit();
                            },
                            _ => {},
                        }
                    },
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    } => {
                        if *state == ElementState::Pressed {
                            input_state.action_triggered = Some(UserAction::DestroyVoxel);
                        }
                    },
                    _ => {},
                }
            },

            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, .. }, ..
            } => {
                input_state.mouse_dx += delta.0 as f32;
                input_state.mouse_dy += delta.1 as f32;
            },

            Event::AboutToWait => {
                // Update timing
                let now = Instant::now();
                let dt = now.duration_since(*last_frame_time).as_secs_f32();
                *last_frame_time = now;

                // Update FPS counter
                let fps = fps_counter.update(dt);
                if let Some(fps) = fps {
                    window.set_title(&format!("Voxel Engine - FPS: {}", fps));
                }

                // Update and render
                engine.update(dt, &input_state);
                engine.render();

                // Reset input state for next frame
                input_state.reset_mouse_delta();
                input_state.reset_actions();

                // Redraw the window
                window.request_redraw();
            },
            _ => {}
        }
    }).expect("Event loop error");
}

struct FpsCounter {
    frame_count: u32,
    elapsed_time: f32,
    update_interval: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frame_count: 0,
            elapsed_time: 0.0,
            update_interval: 1.0, // Update FPS every second
        }
    }

    fn update(&mut self, dt: f32) -> Option<u32> {
        self.frame_count += 1;
        self.elapsed_time += dt;

        if self.elapsed_time >= self.update_interval {
            let fps = (self.frame_count as f32 / self.elapsed_time).round() as u32;
            self.frame_count = 0;
            self.elapsed_time = 0.0;
            Some(fps)
        } else {
            None
        }
    }
}