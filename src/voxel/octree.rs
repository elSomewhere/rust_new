use glam::IVec3;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

use crate::voxel::types::VoxelData;
use crate::voxel::mesh::MeshStrategy;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OctreeContentType {
    Empty,
    Homogeneous,    // Node filled with same material
    Heterogeneous,  // Node contains multiple materials
    InstanceLeaf,   // Node contains actual instances
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OctreeNode {
    pub center: IVec3,
    pub half_size: i32,
    pub child_mask: u8,
    pub content_type: OctreeContentType,
    pub material_id: Option<u16>,
    pub instance_batch_index: Option<u32>,
    pub mesh_strategy: Option<MeshStrategy>,
    pub children: Option<Box<[Option<OctreeNode>; 8]>>,
}

impl OctreeNode {
    pub fn new(center: IVec3, half_size: i32) -> Self {
        Self {
            center,
            half_size,
            child_mask: 0,
            content_type: OctreeContentType::Empty,
            material_id: None,
            instance_batch_index: None,
            mesh_strategy: None,
            children: None,
        }
    }

    pub fn has_child(&self, index: usize) -> bool {
        self.child_mask & (1 << index) != 0
    }

    pub fn get_child(&self, index: usize) -> Option<&OctreeNode> {
        if !self.has_child(index) {
            return None;
        }

        if let Some(ref children) = self.children {
            return children[index].as_ref();
        }

        None
    }

    pub fn get_child_mut(&mut self, index: usize) -> Option<&mut OctreeNode> {
        if !self.has_child(index) {
            return None;
        }

        if let Some(ref mut children) = self.children {
            return children[index].as_mut();
        }

        None
    }

    pub fn set_child(&mut self, index: usize, node: OctreeNode) {
        // Ensure children array exists
        if self.children.is_none() {
            self.children = Some(Box::new([None, None, None, None, None, None, None, None]));
        }

        // Set the child
        if let Some(ref mut children) = self.children {
            children[index] = Some(node);
            self.child_mask |= 1 << index;
        }
    }

    pub fn remove_child(&mut self, index: usize) {
        if let Some(ref mut children) = self.children {
            children[index] = None;
            self.child_mask &= !(1 << index);

            // If no more children, remove the array
            if self.child_mask == 0 {
                self.children = None;
            }
        }
    }

    pub fn get_child_index_containing_point(&self, point: IVec3) -> usize {
        let mut index = 0;

        if point.x >= self.center.x { index |= 1; }
        if point.y >= self.center.y { index |= 2; }
        if point.z >= self.center.z { index |= 4; }

        index
    }

    pub fn get_child_center(&self, index: usize) -> IVec3 {
        let quarter_size = self.half_size / 2;

        let mut offset = IVec3::ZERO;

        if index & 1 != 0 { offset.x = quarter_size; } else { offset.x = -quarter_size; }
        if index & 2 != 0 { offset.y = quarter_size; } else { offset.y = -quarter_size; }
        if index & 4 != 0 { offset.z = quarter_size; } else { offset.z = -quarter_size; }

        self.center + offset
    }

    pub fn contains_point(&self, point: IVec3) -> bool {
        let min = self.center - IVec3::splat(self.half_size);
        let max = self.center + IVec3::splat(self.half_size);

        point.x >= min.x && point.x < max.x &&
            point.y >= min.y && point.y < max.y &&
            point.z >= min.z && point.z < max.z
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Octree {
    pub root: Option<OctreeNode>,
    pub depth: u32,
}

impl Octree {
    pub fn new() -> Self {
        Self {
            root: None,
            depth: 0,
        }
    }

    pub fn build_from_voxels(&mut self, voxels: &[VoxelData], size: i32) {
        let half_size = size / 2;
        let center = IVec3::splat(half_size);
        let max_depth = (size as f32).log2() as u32;

        self.depth = max_depth;
        self.root = Some(self.build_node(voxels, center, half_size, size, 0, max_depth));
    }

    fn build_node(&self, voxels: &[VoxelData], center: IVec3, half_size: i32,
                  stride: i32, depth: u32, max_depth: u32) -> OctreeNode {
        let mut node = OctreeNode::new(center, half_size);

        if half_size == 0 || depth == max_depth {
            // Leaf node
            let index = self.get_voxel_index(center, stride);
            if index < voxels.len() {
                let voxel = voxels[index];

                if voxel.is_air() {
                    node.content_type = OctreeContentType::Empty;
                } else {
                    node.content_type = OctreeContentType::Homogeneous;
                    node.material_id = Some(voxel.material_id as u16);
                }
            } else {
                node.content_type = OctreeContentType::Empty;
            }

            return node;
        }

        // Check if the node is homogeneous
        let min = center - IVec3::splat(half_size);
        let max = center + IVec3::splat(half_size);

        let mut all_same = true;
        let mut first_material: Option<u8> = None;
        let mut all_empty = true;

        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    let pos = IVec3::new(x, y, z);
                    let index = self.get_voxel_index(pos, stride);

                    if index < voxels.len() {
                        let voxel = voxels[index];

                        if !voxel.is_air() {
                            all_empty = false;

                            if let Some(material) = first_material {
                                if material != voxel.material_id {
                                    all_same = false;
                                    break;
                                }
                            } else {
                                first_material = Some(voxel.material_id);
                            }
                        }
                    }
                }

                if !all_same {
                    break;
                }
            }

            if !all_same {
                break;
            }
        }

        if all_empty {
            node.content_type = OctreeContentType::Empty;
            return node;
        }

        if all_same && first_material.is_some() {
            node.content_type = OctreeContentType::Homogeneous;
            node.material_id = Some(first_material.unwrap() as u16);
            return node;
        }

        // If not homogeneous, create children
        node.content_type = OctreeContentType::Heterogeneous;

        let quarter_size = half_size / 2;
        let new_depth = depth + 1;

        for i in 0..8 {
            let child_center = node.get_child_center(i);

            let child_node = self.build_node(
                voxels,
                child_center,
                quarter_size,
                stride,
                new_depth,
                max_depth
            );

            // Only add non-empty children
            if child_node.content_type != OctreeContentType::Empty {
                node.set_child(i, child_node);
            }
        }

        // Check if all children are the same material
        if node.child_mask != 0 {
            let mut all_children_same = true;
            let mut first_child_material: Option<u16> = None;

            for i in 0..8 {
                if node.has_child(i) {
                    let child = node.get_child(i).unwrap();

                    if child.content_type == OctreeContentType::Homogeneous {
                        if let Some(material) = first_child_material {
                            if material != child.material_id.unwrap() {
                                all_children_same = false;
                                break;
                            }
                        } else {
                            first_child_material = child.material_id;
                        }
                    } else {
                        all_children_same = false;
                        break;
                    }
                }
            }

            // If all children are the same, merge them
            if all_children_same && first_child_material.is_some() {
                node.content_type = OctreeContentType::Homogeneous;
                node.material_id = first_child_material;
                node.children = None;
                node.child_mask = 0;
            }
        }

        node
    }

    fn get_voxel_index(&self, pos: IVec3, stride: i32) -> usize {
        (pos.x + pos.y * stride + pos.z * stride * stride) as usize
    }

    pub fn update_voxel(&mut self, pos: IVec3, voxel: VoxelData) {
        // Create the root if it doesn't exist
        if self.root.is_none() {
            return;
        }

        if let Some(ref mut root) = self.root {
            // If the voxel is outside the octree, ignore
            if !root.contains_point(pos) {
                return;
            }

            // IMPORTANT FIX: Store the depth value before the recursive call
            // to avoid the double borrow of self
            let depth = self.depth;

            // Making update_node_voxel_recursive a static function to avoid borrowing self
            Self::update_node_voxel_recursive(root, pos, voxel, depth);
        }
    }


    // Changed from a method to a static function (no &mut self parameter)
    fn update_node_voxel_recursive(node: &mut OctreeNode, pos: IVec3, voxel: VoxelData, depth: u32) {
        if depth == 0 || node.half_size <= 1 {
            // Leaf node, update directly
            if voxel.is_air() {
                node.content_type = OctreeContentType::Empty;
                node.material_id = None;
            } else {
                node.content_type = OctreeContentType::Homogeneous;
                node.material_id = Some(voxel.material_id as u16);
            }
            return;
        }

        // Rest of the function remains the same, just remove any self. references
        // Get the child index for this position
        let child_index = node.get_child_index_containing_point(pos);

        // If the node is homogeneous and we're making it heterogeneous
        if node.content_type == OctreeContentType::Homogeneous &&
            (node.material_id.unwrap() as u8 != voxel.material_id ||
                (node.material_id.unwrap() == 0 && !voxel.is_air()) ||
                (node.material_id.unwrap() != 0 && voxel.is_air())) {

            // Split the node
            node.content_type = OctreeContentType::Heterogeneous;
            let current_material = node.material_id.unwrap();

            // Create children with the current material
            for i in 0..8 {
                let child_center = node.get_child_center(i);
                let mut child = OctreeNode::new(child_center, node.half_size / 2);

                if current_material == 0 {
                    child.content_type = OctreeContentType::Empty;
                } else {
                    child.content_type = OctreeContentType::Homogeneous;
                    child.material_id = Some(current_material);
                }

                node.set_child(i, child);
            }
        }

        // If node is heterogeneous, update the child
        if node.content_type == OctreeContentType::Heterogeneous {
            // Create the child if it doesn't exist
            if !node.has_child(child_index) {
                let child_center = node.get_child_center(child_index);
                let child = OctreeNode::new(child_center, node.half_size / 2);
                node.set_child(child_index, child);
            }

            // Update recursively
            if let Some(child) = node.get_child_mut(child_index) {
                Self::update_node_voxel_recursive(child, pos, voxel, depth - 1);
            }

            // Check if children can be merged
            let mut all_children_same = true;
            let mut first_material: Option<u16> = None;
            let mut all_empty = true;

            for i in 0..8 {
                if node.has_child(i) {
                    if let Some(child) = node.get_child(i) {
                        if child.content_type == OctreeContentType::Empty {
                            if first_material.is_some() {
                                all_children_same = false;
                                break;
                            }
                        } else if child.content_type == OctreeContentType::Homogeneous {
                            all_empty = false;

                            if let Some(material) = first_material {
                                if material != child.material_id.unwrap() {
                                    all_children_same = false;
                                    break;
                                }
                            } else {
                                first_material = child.material_id;
                            }
                        } else {
                            all_children_same = false;
                            all_empty = false;
                            break;
                        }
                    }
                }
            }

            // If all children can be merged
            if all_children_same {
                if all_empty {
                    node.content_type = OctreeContentType::Empty;
                    node.material_id = None;
                } else if first_material.is_some() {
                    node.content_type = OctreeContentType::Homogeneous;
                    node.material_id = first_material;
                }

                node.children = None;
                node.child_mask = 0;
            }
        }
    }

    pub fn iterate_leaf_nodes<F>(&self, mut f: F)
    where
        F: FnMut(&OctreeNode),
    {
        if let Some(ref root) = self.root {
            self.iterate_node(root, &mut f);
        }
    }

    fn iterate_node<F>(&self, node: &OctreeNode, f: &mut F)
    where
        F: FnMut(&OctreeNode),
    {
        // Process this node if it's a leaf or homogeneous
        if node.content_type == OctreeContentType::Homogeneous ||
            node.content_type == OctreeContentType::InstanceLeaf ||
            node.child_mask == 0 {
            f(node);
            return;
        }

        // Otherwise, process children
        for i in 0..8 {
            if node.has_child(i) {
                if let Some(child) = node.get_child(i) {
                    self.iterate_node(child, f);
                }
            }
        }
    }

    pub fn get_node_at_position<'a>(&'a self, pos: IVec3) -> Option<&'a OctreeNode> {
        if let Some(ref root) = self.root {
            if !root.contains_point(pos) {
                return None;
            }

            return self.get_node_at_position_recursive(root, pos, self.depth);
        }

        None
    }

    fn get_node_at_position_recursive<'a>(&'a self, node: &'a OctreeNode, pos: IVec3, depth: u32) -> Option<&'a OctreeNode> {
        // If homogeneous or leaf node, return this node
        if node.content_type == OctreeContentType::Homogeneous ||
            node.content_type == OctreeContentType::InstanceLeaf ||
            depth == 0 || node.child_mask == 0 {
            return Some(node);
        }

        // Otherwise, recurse into child
        let child_index = node.get_child_index_containing_point(pos);

        if node.has_child(child_index) {
            if let Some(child) = node.get_child(child_index) {
                return self.get_node_at_position_recursive(child, pos, depth - 1);
            }
        }

        // If no child exists at this position, return this node
        Some(node)
    }
}