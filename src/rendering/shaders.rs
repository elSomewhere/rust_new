// This module contains shader code

// Instance rendering shader
const INSTANCE_SHADER: &str = r#"
// Vertex shader
struct Camera {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    position: vec4<f32>,
};

struct Material {
    albedo_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emission_strength: f32,
    flags: u32,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var<uniform> materials: array<Material, 16>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct InstanceInput {
    @location(5) position: vec3<f32>,
    @location(6) rotation: u32,
    @location(7) scale: f32,
    @location(8) material_index: u32,
    @location(9) ao_data: u32,
    @location(10) custom_data: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) material_index: u32,
    @location(5) ao_value: f32,
};

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;

    // Apply scale to the vertex position
    let scaled_position = vertex.position * instance.scale;

    // Add instance position
    let world_position = scaled_position + instance.position;

    // Pass data to fragment shader
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.position = world_position;
    out.normal = vertex.normal;
    out.tex_coords = vertex.tex_coords;
    out.color = vertex.color;
    out.material_index = instance.material_index;

    // Calculate AO value (0-1)
    out.ao_value = f32(instance.ao_data) / 255.0;

    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Get the material index from the color's alpha channel
    let material_index = u32(in.color.a * 255.0);

    // Debug mode: Visualize normal directions with color
    let debug_normals = false; // Set to false for normal rendering
    if (debug_normals) {
        // Map normal direction to RGB color for visualization
        let normal_color = (normalize(in.normal) + vec3<f32>(1.0, 1.0, 1.0)) * 0.5;
        return vec4<f32>(normal_color, 1.0);
    }

    // Ambient lighting - increased for better visibility
    let ambient = 0.4;

    // Directional light
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.5));
    let normal = normalize(in.normal);

    // Calculate diffuse lighting - ensure we're using the correct normal
    let diffuse = max(dot(normal, light_dir), 0.0);

    // Calculate final color
    let light_color = vec3<f32>(1.0, 0.95, 0.9);

    // Increase the diffuse factor for better visibility
    let light_intensity = ambient + diffuse * 0.8;

    // Use the color RGB components directly
    let final_color = in.color.rgb * light_color * light_intensity;

    return vec4<f32>(final_color, in.color.a);
}
"#;

// Mesh rendering shader
const MESH_SHADER: &str = r#"
// Vertex shader
struct Camera {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    position: vec4<f32>,
};

struct Material {
    albedo_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emission_strength: f32,
    flags: u32,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var<uniform> materials: array<Material, 16>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(
    vertex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    // Transform vertex
    out.clip_position = camera.view_proj * vec4<f32>(vertex.position, 1.0);
    out.position = vertex.position;
    out.normal = vertex.normal;
    out.tex_coords = vertex.tex_coords;
    out.color = vertex.color;

    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Get the material index from the color's alpha channel (hack for this demo)
    let material_index = u32(in.color.a * 255.0);

    // Ambient lighting
    let ambient = 0.3;

    // Directional light
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.5));
    let normal = normalize(in.normal);
    let diffuse = max(dot(normal, light_dir), 0.0);

    // Calculate final color
    let light_color = vec3<f32>(1.0, 0.95, 0.9);
    let light_intensity = ambient + diffuse * 0.7;
    let final_color = in.color.rgb * light_color * light_intensity;

    return vec4<f32>(final_color, in.color.a);
}
"#;

// Load instance shader
pub fn get_instance_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Instance Shader"),
        source: wgpu::ShaderSource::Wgsl(INSTANCE_SHADER.into()),
    })
}

// Load mesh shader
pub fn get_mesh_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mesh Shader"),
        source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
    })
}