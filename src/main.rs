use std::time::Instant;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{WindowBuilder, Window},
    keyboard::{KeyCode, PhysicalKey},
};

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

struct GameState {
    engine: Engine,
    input_state: InputState,
    last_frame_time: Instant,
    fps_counter: FpsCounter,
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let window = Box::leak(Box::new(
        WindowBuilder::new()
            .with_title("Voxel Engine")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
            .build(&event_loop)
            .expect("Failed to create window"),
    ));

    let window_id = window.id();

    window
        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
        .expect("Failed to grab cursor");
    window.set_cursor_visible(false);

    // Initialize engine with the window directly
    let engine = pollster::block_on(Engine::new(window));

    let mut game_state = GameState {
        engine,
        input_state: InputState::default(),
        last_frame_time: Instant::now(),
        fps_counter: FpsCounter::new(),
    };

    let window_ref = window as &'static Window;

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id: event_window_id,
            } if event_window_id == window_id => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    game_state.engine.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    game_state.engine.resize(window_ref.inner_size());
                }
                WindowEvent::KeyboardInput {
                    event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                    ..
                } => {
                    let pressed = state == &ElementState::Pressed;
                    match keycode {
                        KeyCode::KeyW | KeyCode::ArrowUp => {
                            game_state.input_state.is_forward_pressed = pressed;
                        }
                        KeyCode::KeyS | KeyCode::ArrowDown => {
                            game_state.input_state.is_backward_pressed = pressed;
                        }
                        KeyCode::KeyA | KeyCode::ArrowLeft => {
                            game_state.input_state.is_left_pressed = pressed;
                        }
                        KeyCode::KeyD | KeyCode::ArrowRight => {
                            game_state.input_state.is_right_pressed = pressed;
                        }
                        KeyCode::Space => {
                            if pressed {
                                game_state.input_state.action_triggered =
                                    Some(UserAction::CreateExplosion);
                            } else {
                                game_state.input_state.is_up_pressed = false;
                            }
                        }
                        KeyCode::ShiftLeft => {
                            game_state.input_state.is_run_pressed = pressed;
                        }
                        KeyCode::ControlLeft => {
                            game_state.input_state.is_down_pressed = pressed;
                        }
                        KeyCode::Digit1 => {
                            if pressed {
                                game_state.input_state.action_triggered =
                                    Some(UserAction::ChangeStrategy(MeshStrategy::Instanced));
                            }
                        }
                        KeyCode::Digit2 => {
                            if pressed {
                                game_state.input_state.action_triggered =
                                    Some(UserAction::ChangeStrategy(MeshStrategy::GreedyMesh));
                            }
                        }
                        KeyCode::Digit3 => {
                            if pressed {
                                game_state.input_state.action_triggered =
                                    Some(UserAction::ChangeStrategy(MeshStrategy::MarchingCubes));
                            }
                        }
                        KeyCode::Digit4 => {
                            if pressed {
                                game_state.input_state.action_triggered =
                                    Some(UserAction::ChangeStrategy(MeshStrategy::DualContouring));
                            }
                        }
                        KeyCode::Escape => {
                            elwt.exit();
                        }
                        _ => {}
                    }
                }
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    if *state == ElementState::Pressed {
                        game_state.input_state.action_triggered = Some(UserAction::DestroyVoxel);
                    }
                }
                _ => {}
            },

            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta, .. },
                ..
            } => {
                game_state.input_state.mouse_dx += delta.0 as f32;
                game_state.input_state.mouse_dy += delta.1 as f32;
            }

            Event::AboutToWait => {
                let now = Instant::now();
                let dt = now.duration_since(game_state.last_frame_time).as_secs_f32();
                game_state.last_frame_time = now;

                let fps = game_state.fps_counter.update(dt);
                if let Some(fps) = fps {
                    window_ref.set_title(&format!("Voxel Engine - FPS: {}", fps));
                }

                game_state.engine.update(dt, &game_state.input_state);
                game_state.engine.render();

                game_state.input_state.reset_mouse_delta();
                game_state.input_state.reset_actions();

                window_ref.request_redraw();
            }
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
            update_interval: 1.0,
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