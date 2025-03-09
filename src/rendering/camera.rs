use glam::{Vec3, Mat4, Quat};
use bytemuck::{Pod, Zeroable};

use crate::InputState;

#[derive(Debug)]
pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub aspect: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(position: Vec3, yaw: f32, pitch: f32, aspect: f32) -> Self {
        Self {
            position,
            yaw,
            pitch,
            aspect,
            fov: 60.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }

    pub fn build_view_matrix(&self) -> Mat4 {
        // Create rotation quaternion from yaw and pitch
        let quat = Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch);

        // Convert to 3x3 rotation matrix
        let rotation = Mat4::from_quat(quat);

        // Create translation matrix
        let translation = Mat4::from_translation(-self.position);

        // Combine rotation and translation
        rotation * translation
    }

    pub fn build_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = self.build_view_matrix();
        let proj = self.build_projection_matrix();
        proj * view
    }

    pub fn get_view_direction(&self) -> Vec3 {
        // Calculate the direction vector based on yaw and pitch
        let x = self.yaw.cos() * self.pitch.cos();
        let y = self.pitch.sin();
        let z = self.yaw.sin() * self.pitch.cos();

        Vec3::new(x, y, z).normalize()
    }

    pub fn get_right_vector(&self) -> Vec3 {
        // Right vector is perpendicular to view direction and up
        let forward = self.get_view_direction();
        let up = Vec3::new(0.0, 1.0, 0.0);
        forward.cross(up).normalize()
    }

    pub fn get_up_vector(&self) -> Vec3 {
        // Up vector is perpendicular to view direction and right
        let forward = self.get_view_direction();
        let right = self.get_right_vector();
        right.cross(forward).normalize()
    }
}

// Controller for moving the camera
pub struct CameraController {
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera, input: &InputState, dt: f32) {
        // Handle mouse input for rotation
        let dx = input.mouse_dx;
        let dy = input.mouse_dy;

        camera.yaw += dx * self.sensitivity * dt;
        camera.pitch = (camera.pitch - dy * self.sensitivity * dt)
            .clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);

        // Calculate movement directions
        let forward = Vec3::new(
            camera.yaw.cos(),
            0.0,
            camera.yaw.sin(),
        ).normalize();

        let right = Vec3::new(
            camera.yaw.sin(),
            0.0,
            -camera.yaw.cos(),
        ).normalize();

        // Calculate movement speed
        let mut actual_speed = self.speed;
        if input.is_run_pressed {
            actual_speed *= 3.0; // Sprint multiplier
        }

        // Apply movement
        let mut velocity = Vec3::ZERO;

        if input.is_forward_pressed {
            velocity += forward;
        }
        if input.is_backward_pressed {
            velocity -= forward;
        }
        if input.is_right_pressed {
            velocity += right;
        }
        if input.is_left_pressed {
            velocity -= right;
        }
        if input.is_up_pressed {
            velocity += Vec3::Y;
        }
        if input.is_down_pressed {
            velocity -= Vec3::Y;
        }

        // Normalize velocity to keep diagonal movement at same speed
        if velocity != Vec3::ZERO {
            velocity = velocity.normalize();
            camera.position += velocity * actual_speed * dt;
        }
    }
}

// Uniform buffer for passing camera data to shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub position: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            projection: Mat4::IDENTITY.to_cols_array_2d(),
            position: [0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view = camera.build_view_matrix().to_cols_array_2d();
        self.projection = camera.build_projection_matrix().to_cols_array_2d();
        self.view_proj = camera.build_view_projection_matrix().to_cols_array_2d();
        self.position = [camera.position.x, camera.position.y, camera.position.z, 1.0];
    }
}