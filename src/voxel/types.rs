use serde::{Serialize, Deserialize};
use glam::{Vec3, Vec4};

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct VoxelData {
    pub material_id: u8,
    pub density: u8,
    pub temperature: u8,
    pub moisture: u8,
}

impl VoxelData {
    pub fn new(material_id: u8, density: u8, temperature: u8, moisture: u8) -> Self {
        Self {
            material_id,
            density,
            temperature,
            moisture,
        }
    }

    pub fn air() -> Self {
        Self {
            material_id: 0,
            density: 0,
            temperature: 128,
            moisture: 0,
        }
    }

    pub fn stone() -> Self {
        Self {
            material_id: 1,
            density: 255,
            temperature: 128,
            moisture: 0,
        }
    }

    pub fn dirt() -> Self {
        Self {
            material_id: 2,
            density: 255,
            temperature: 128,
            moisture: 128,
        }
    }

    pub fn grass() -> Self {
        Self {
            material_id: 3,
            density: 255,
            temperature: 128,
            moisture: 192,
        }
    }

    pub fn sand() -> Self {
        Self {
            material_id: 4,
            density: 255,
            temperature: 192,
            moisture: 64,
        }
    }

    pub fn water() -> Self {
        Self {
            material_id: 5,
            density: 128,
            temperature: 128,
            moisture: 255,
        }
    }

    pub fn is_air(&self) -> bool {
        self.material_id == 0
    }

    pub fn is_solid(&self) -> bool {
        self.material_id != 0 && self.density >= 128
    }

    pub fn is_transparent(&self) -> bool {
        self.material_id == 0 || self.material_id == 5 // Air or water
    }

    pub fn is_fluid(&self) -> bool {
        self.material_id == 5 // Water
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Material {
    pub albedo_color: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub emission: Vec3,
    pub flags: u32,
    pub rendering_mode: RenderingMode,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RenderingMode {
    Blocky = 0,
    Smooth = 1,
    Auto = 2,
    GpuDriven = 3,
}

impl Material {
    pub fn new(
        albedo_color: Vec4,
        metallic: f32,
        roughness: f32,
        emission: Vec3,
        flags: u32,
        rendering_mode: RenderingMode,
    ) -> Self {
        Self {
            albedo_color,
            metallic,
            roughness,
            emission,
            flags,
            rendering_mode,
        }
    }

    pub fn default_materials() -> Vec<Material> {
        // Create base materials
        let base_materials = vec![
            // 0: Air
            Material::new(
                Vec4::new(1.0, 1.0, 1.0, 0.0),
                0.0,
                1.0,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Auto,
            ),
            // 1: Stone
            Material::new(
                Vec4::new(0.5, 0.5, 0.5, 1.0),
                0.0,
                0.8,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 2: Dirt
            Material::new(
                Vec4::new(0.6, 0.3, 0.1, 1.0),
                0.0,
                0.9,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 3: Grass
            Material::new(
                Vec4::new(0.3, 0.7, 0.2, 1.0),
                0.0,
                0.9,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 4: Sand
            Material::new(
                Vec4::new(0.9, 0.8, 0.6, 1.0),
                0.0,
                0.8,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 5: Water
            Material::new(
                Vec4::new(0.2, 0.4, 0.8, 0.7),
                0.0,
                0.4,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Smooth,
            ),
            // 6: Wood
            Material::new(
                Vec4::new(0.5, 0.3, 0.1, 1.0),
                0.0,
                0.7,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 7: Leaves
            Material::new(
                Vec4::new(0.2, 0.5, 0.1, 0.9),
                0.0,
                0.9,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Auto,
            ),
            // 8: Gold Ore
            Material::new(
                Vec4::new(0.9, 0.8, 0.1, 1.0),
                0.7,
                0.4,
                Vec3::new(0.2, 0.2, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 9: Iron Ore
            Material::new(
                Vec4::new(0.6, 0.5, 0.5, 1.0),
                0.3,
                0.7,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 10: Coal Ore
            Material::new(
                Vec4::new(0.2, 0.2, 0.2, 1.0),
                0.0,
                0.8,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 11: Bedrock
            Material::new(
                Vec4::new(0.3, 0.3, 0.3, 1.0),
                0.0,
                1.0,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 12: Lava
            Material::new(
                Vec4::new(0.9, 0.3, 0.0, 0.9),
                0.0,
                0.3,
                Vec3::new(0.8, 0.3, 0.0),
                0,
                RenderingMode::Smooth,
            ),
            // 13: Snow
            Material::new(
                Vec4::new(0.95, 0.95, 0.95, 1.0),
                0.0,
                0.95,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Blocky,
            ),
            // 14: Ice
            Material::new(
                Vec4::new(0.8, 0.9, 1.0, 0.8),
                0.1,
                0.1,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Smooth,
            ),
            // 15: Glass
            Material::new(
                Vec4::new(0.9, 0.9, 0.9, 0.4),
                0.5,
                0.1,
                Vec3::new(0.0, 0.0, 0.0),
                0,
                RenderingMode::Smooth,
            ),
        ];

        // Ensure we have exactly 16 materials
        assert_eq!(base_materials.len(), 16, "Materials array must have exactly 16 elements");

        base_materials
    }
}

// Instance data structure for rendering
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct VoxelInstance {
    pub position: Vec3,
    pub rotation: u8,    // Packed rotation index (24 possible orientations)
    pub scale: f32,
    pub material_index: u16,
    pub ao_data: u8,     // Packed ambient occlusion data
    pub custom_data: u16, // For special effects or behaviors
}

impl VoxelInstance {
    pub fn new(
        position: Vec3,
        rotation: u8,
        scale: f32,
        material_index: u16,
        ao_data: u8,
        custom_data: u16,
    ) -> Self {
        Self {
            position,
            rotation,
            scale,
            material_index,
            ao_data,
            custom_data,
        }
    }
}

// Ensure VoxelInstance can be used in GPU buffers
unsafe impl bytemuck::Pod for VoxelInstance {}
unsafe impl bytemuck::Zeroable for VoxelInstance {}

// Draw command for batched rendering
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

// Ensure DrawIndirectCommand can be used in GPU buffers
unsafe impl bytemuck::Pod for DrawIndirectCommand {}
unsafe impl bytemuck::Zeroable for DrawIndirectCommand {}