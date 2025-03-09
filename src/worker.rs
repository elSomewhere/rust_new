use std::collections::{VecDeque, HashMap};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use log::{debug, warn, error};
use glam::IVec3;
use crossbeam_channel::{bounded, Sender, Receiver, RecvTimeoutError};
use parking_lot::RwLock;

use crate::voxel::types::VoxelData;
use crate::voxel::octree::Octree;
use crate::voxel::mesh::{MeshStrategy, MeshData, MeshGenerator};
use crate::voxel::procedural::TerrainGenerator;
use crate::voxel::CHUNK_SIZE;

// Define a task enum to represent different kinds of work
#[derive(Debug)]
pub enum WorkerTask {
    GenerateChunk {
        position: IVec3,
    },
    GenerateMesh {
        chunk_position: IVec3,
        strategy: MeshStrategy,
    },
    SaveChunk {
        position: IVec3,
        voxels: Vec<VoxelData>,
    },
    LoadChunk {
        position: IVec3,
    },
    // Add other task types as needed
}

// Define results for completed tasks
#[derive(Debug)]
pub enum WorkerTaskResult {
    ChunkGenerated {
        position: IVec3,
        voxels: Vec<VoxelData>,
        octree: Octree,
    },
    MeshGenerated {
        position: IVec3,
        strategy: MeshStrategy,
        mesh_data: MeshData,
    },
    ChunkSaved {
        position: IVec3,
    },
    ChunkLoaded {
        position: IVec3,
        voxels: Vec<VoxelData>,
    },
    Error {
        task: Box<WorkerTask>,
        message: String,
    },
}

// Worker thread state
struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
    is_busy: Arc<RwLock<bool>>,
}

// WorkerSystem manages a pool of worker threads
pub struct WorkerSystem {
    workers: Vec<Worker>,
    task_sender: Sender<WorkerTask>,
    result_receiver: Receiver<WorkerTaskResult>,
    pending_results: VecDeque<WorkerTaskResult>,
    task_count: usize,
    terrain_generator: Arc<TerrainGenerator>,
    chunk_cache: HashMap<IVec3, Vec<VoxelData>>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl WorkerSystem {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        // Create channels for communication
        let (task_sender, task_receiver) = bounded(100);
        let (result_sender, result_receiver) = bounded(100);

        // Create a terrain generator
        let terrain_generator = Arc::new(TerrainGenerator::new(42));

        // Determine number of worker threads (one less than available, minimum 1)
        let num_workers = ((num_cpus::get() as i32) - 1).max(1) as usize;

        let mut workers = Vec::with_capacity(num_workers);

        for id in 0..num_workers {
            let task_receiver = task_receiver.clone();
            let result_sender = result_sender.clone();
            let is_busy = Arc::new(RwLock::new(false));
            let is_busy_clone = Arc::clone(&is_busy);
            let terrain_generator = Arc::clone(&terrain_generator);

            // Spawn worker thread
            let thread = thread::spawn(move || {
                Self::worker_thread(
                    id,
                    task_receiver,
                    result_sender,
                    is_busy_clone,
                    terrain_generator,
                );
            });

            workers.push(Worker {
                id,
                thread: Some(thread),
                is_busy,
            });
        }

        Self {
            workers,
            task_sender,
            result_receiver,
            pending_results: VecDeque::new(),
            task_count: 0,
            terrain_generator,
            chunk_cache: HashMap::new(),
            device,
            queue,
        }
    }

    // Worker thread function
    fn worker_thread(
        id: usize,
        task_receiver: Receiver<WorkerTask>,
        result_sender: Sender<WorkerTaskResult>,
        is_busy: Arc<RwLock<bool>>,
        terrain_generator: Arc<TerrainGenerator>,
    ) {
        debug!("Worker {} started", id);

        loop {
            // Try to receive a task with timeout
            match task_receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(task) => {
                    // Mark worker as busy
                    *is_busy.write() = true;

                    // Process the task
                    let result = match task {
                        WorkerTask::GenerateChunk { position } => {
                            debug!("Worker {} generating chunk at {:?}", id, position);

                            // Generate the chunk
                            let (voxels, octree) = terrain_generator.generate_chunk(position);

                            // Return the result
                            WorkerTaskResult::ChunkGenerated {
                                position,
                                voxels,
                                octree,
                            }
                        },
                        WorkerTask::GenerateMesh { chunk_position, strategy } => {
                            debug!("Worker {} generating mesh for chunk {:?} with strategy {:?}",
                                   id, chunk_position, strategy);

                            // Generate terrain data if it's not already available
                            let (voxels, _) = terrain_generator.generate_chunk(chunk_position);

                            // Generate mesh based on the strategy
                            let mesh_data = match strategy {
                                MeshStrategy::Instanced =>
                                    MeshGenerator::generate_instanced_mesh(&voxels, chunk_position, CHUNK_SIZE),
                                MeshStrategy::GreedyMesh =>
                                    MeshGenerator::generate_greedy_mesh(&voxels, chunk_position, CHUNK_SIZE),
                                MeshStrategy::MarchingCubes =>
                                    MeshGenerator::generate_marching_cubes_mesh(&voxels, chunk_position, CHUNK_SIZE),
                                MeshStrategy::DualContouring =>
                                    MeshGenerator::generate_dual_contouring_mesh(&voxels, chunk_position, CHUNK_SIZE),
                                MeshStrategy::MeshShader => {
                                    // Mesh shaders would be implemented in the GPU,
                                    // but we'll fall back to instanced for now
                                    MeshGenerator::generate_instanced_mesh(&voxels, chunk_position, CHUNK_SIZE)
                                },
                            };

                            // Return the result
                            WorkerTaskResult::MeshGenerated {
                                position: chunk_position,
                                strategy,
                                mesh_data,
                            }
                        },
                        WorkerTask::SaveChunk { position, voxels } => {
                            debug!("Worker {} saving chunk at {:?}", id, position);

                            // In a real implementation, this would save to disk
                            // For this example, we'll just simulate the operation
                            thread::sleep(Duration::from_millis(10));

                            WorkerTaskResult::ChunkSaved {
                                position,
                            }
                        },
                        WorkerTask::LoadChunk { position } => {
                            debug!("Worker {} loading chunk at {:?}", id, position);

                            // In a real implementation, this would load from disk
                            // For this example, we'll generate a new chunk
                            let (voxels, _) = terrain_generator.generate_chunk(position);

                            WorkerTaskResult::ChunkLoaded {
                                position,
                                voxels,
                            }
                        },
                    };

                    // Send result back
                    match result_sender.send(result) {
                        Ok(_) => {},
                        Err(e) => error!("Worker {} failed to send result: {:?}", id, e),
                    }

                    // Mark worker as free
                    *is_busy.write() = false;
                },
                Err(RecvTimeoutError::Timeout) => {
                    // Timeout - just continue waiting
                },
                Err(RecvTimeoutError::Disconnected) => {
                    // Channel closed - exit thread
                    debug!("Worker {} exiting - channel closed", id);
                    break;
                }
            }
        }
    }

    // Submit a task to the worker system
    pub fn submit_task(&mut self, task: WorkerTask) {
        match self.task_sender.send(task) {
            Ok(_) => {
                self.task_count += 1;
            },
            Err(e) => {
                error!("Failed to submit task: {:?}", e);
            }
        }
    }

    // Update function to check for completed tasks
    pub fn update(&mut self) {
        // Check for new results
        while let Ok(result) = self.result_receiver.try_recv() {
            self.pending_results.push_back(result);
            self.task_count -= 1;
        }

        // Implement work stealing if needed
        self.balance_workload();
    }

    // Balance workload among workers (work stealing)
    fn balance_workload(&mut self) {
        // Simplified work stealing implementation
        // In a real system, this would be more sophisticated

        // Count free workers vs busy workers
        let mut free_count = 0;
        let mut busy_count = 0;

        for worker in &self.workers {
            if *worker.is_busy.read() {
                busy_count += 1;
            } else {
                free_count += 1;
            }
        }

        // Log worker status if there's a significant imbalance
        if busy_count > 0 && free_count == 0 {
            debug!("All workers busy: {} tasks remaining", self.task_count);
        } else if busy_count == 0 && free_count > 0 && self.task_count > 0 {
            warn!("All workers free but tasks remain: {}", self.task_count);
        }

        // In a real implementation, we would have logic to redistribute tasks
    }

    // Process results with a callback
    pub fn handle_completed_tasks<F>(&mut self, mut callback: F)
    where
        F: FnMut(WorkerTaskResult),
    {
        while let Some(result) = self.pending_results.pop_front() {
            // Cache chunks for mesh generation
            match &result {
                WorkerTaskResult::ChunkGenerated { position, voxels, .. } => {
                    self.chunk_cache.insert(*position, voxels.clone());
                },
                WorkerTaskResult::ChunkLoaded { position, voxels } => {
                    self.chunk_cache.insert(*position, voxels.clone());
                },
                _ => {},
            }

            callback(result);
        }
    }

    // Get a cached chunk
    pub fn get_cached_chunk(&self, position: IVec3) -> Option<&Vec<VoxelData>> {
        self.chunk_cache.get(&position)
    }
}

// Implementation of shutdown in Drop to clean up threads
impl Drop for WorkerSystem {
    fn drop(&mut self) {
        // Close channels to signal threads to exit
        drop(self.task_sender.clone());

        // Wait for threads to finish
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }

        debug!("WorkerSystem shut down, all threads joined");
    }
}