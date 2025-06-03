#![allow(unused_imports)]

use anyhow::Result;
use eframe::egui::{self, ViewportBuilder, ViewportCommand};
use image::DynamicImage;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use walkdir::WalkDir;
use std::fs;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use ndarray::{Array1, Array2, Array4, CowArray};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use sha2::{Sha256, Digest};
use hex;
use rand::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use eframe::epaint::CornerRadius;
use rand::rng;
use ort::{Environment, SessionBuilder, Value};

#[cfg(feature = "gpu")]
use candle_core::{Device as CandleDevice, Tensor as CandleTensor, DType};
#[cfg(feature = "gpu")]
use candle_nn::{Linear, Module, VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};

// Function to find model file relative to executable
fn find_model_path() -> Result<PathBuf> {
    // First try relative to current directory (for cargo run)
    let current_dir_model = PathBuf::from("models/resnet50-v1-7-compatible.onnx");
    if current_dir_model.exists() {
        return Ok(current_dir_model);
    }

    // Then try relative to executable (for packaged version)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let exe_dir_model = exe_dir.join("models/resnet50-v1-7-compatible.onnx");
            if exe_dir_model.exists() {
                return Ok(exe_dir_model);
            }

            // Try one level up from executable (common packaging structure)
            let parent_dir_model = exe_dir.parent()
                .map(|p| p.join("models/resnet50-v1-7-compatible.onnx"));
            if let Some(ref path) = parent_dir_model {
                if path.exists() {
                    return Ok(path.clone());
                }
            }
        }
    }

    anyhow::bail!("Could not find model file 'resnet50-v1-7-compatible.onnx'. Please ensure it exists in:\n\
                   - ./models/ (relative to current directory)\n\
                   - <executable_dir>/models/\n\
                   - <executable_parent_dir>/models/")
}

// Static flag to ensure GPU initialization message is only printed once
#[allow(dead_code)]
static GPU_MESSAGE_PRINTED: AtomicBool = AtomicBool::new(false);

// Image processing thread communication types
#[derive(Clone)]
#[allow(dead_code)]
enum ImageProcessingRequest {
    LoadImage { index: usize, path: PathBuf },
    #[allow(dead_code)]
    PrepareTexture { index: usize, image: DynamicImage, display_size: (u32, u32) },
    ExtractFeatures { hash: String, image: DynamicImage },
    #[allow(dead_code)]
    ComputeHash { index: usize, image: DynamicImage },
}

#[derive(Clone)]
#[allow(dead_code)]
enum ImageProcessingResponse {
    ImageLoaded { index: usize, image: DynamicImage },
    #[allow(dead_code)]
    TexturePrepared { index: usize, rgba_data: Vec<u8>, width: usize, height: usize },
    FeaturesExtracted { hash: String, features: Vec<f32> },
    #[allow(dead_code)]
    HashComputed { index: usize, hash: String },
}

// GPU-optimized image processing
struct OptimizedImage {
    original: DynamicImage,
    display_texture: Option<egui::TextureHandle>,
    display_size: (u32, u32),
    needs_resize: bool,
}

impl OptimizedImage {
    fn new(image: DynamicImage) -> Self {
        // Pre-calculate if we need to resize for performance
        let (w, h) = (image.width(), image.height());
        let needs_resize = w > 2048 || h > 2048; // Resize images larger than 2K for better performance

        Self {
            display_size: if needs_resize {
                // Calculate optimal display size maintaining aspect ratio
                let max_dim = 2048.0;
                let aspect = w as f32 / h as f32;
                if w > h {
                    ((max_dim as u32), (max_dim / aspect) as u32)
                } else {
                    ((max_dim * aspect) as u32, max_dim as u32)
                }
            } else {
                (w, h)
            },
            original: image,
            display_texture: None,
            needs_resize,
        }
    }

    fn get_display_image(&self) -> DynamicImage {
        if self.needs_resize {
            self.original.resize(
                self.display_size.0,
                self.display_size.1,
                image::imageops::FilterType::Lanczos3, // High quality resizing
            )
        } else {
            self.original.clone()
        }
    }

    fn create_optimized_texture(&mut self, ctx: &egui::Context, key: String) -> &egui::TextureHandle {
        if self.display_texture.is_none() {
            let display_img = self.get_display_image();
            let rgba_data = display_img.to_rgba8();

            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                [display_img.width() as usize, display_img.height() as usize],
                &rgba_data.into_raw(),
            );

            // Use simpler texture options during heavy usage to reduce GPU load
            let texture_options = egui::TextureOptions {
                magnification: egui::TextureFilter::Nearest, // Faster than Linear during window movement
                minification: egui::TextureFilter::Linear,
                wrap_mode: egui::TextureWrapMode::ClampToEdge,
                mipmap_mode: None,
            };

            self.display_texture = Some(ctx.load_texture(key, color_image, texture_options));
        }

        self.display_texture.as_ref().unwrap()
    }
}

#[derive(Serialize, Deserialize, Default)]
struct AppState {
    last_positions: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ImageRating {
    image_hash: String,
    rating: u8,
    features: Vec<f32>,
}

// Animation system for smooth effects
#[derive(Clone)]
struct FlashAnimation {
    start_time: Instant,
    duration: Duration,
    color: egui::Color32,
    active: bool,
}

impl FlashAnimation {
    fn new(color: egui::Color32) -> Self {
        Self {
            start_time: Instant::now(),
            duration: Duration::from_millis(400), // Longer duration so users can actually see it
            color,
            active: true,
        }
    }

    fn update(&mut self) -> f32 {
        if !self.active {
            return 0.0;
        }

        let elapsed = self.start_time.elapsed();
        if elapsed >= self.duration {
            self.active = false;
            return 0.0;
        }

        // Stronger easing function for better visibility
        let progress = elapsed.as_secs_f32() / self.duration.as_secs_f32();
        let intensity = 1.0 - progress;
        // Use quadratic easing for more visible effect
        intensity * intensity
    }

    fn is_active(&self) -> bool {
        self.active
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.active = false;
    }
}

// Optimized texture cache with GPU acceleration focus
struct TextureCache {
    images: HashMap<String, OptimizedImage>,
    last_used: HashMap<String, Instant>,
    max_cache_size: usize,
    total_memory_estimate: usize, // Track approximate memory usage
    max_memory_mb: usize,
}

impl TextureCache {
    fn new() -> Self {
        Self {
            images: HashMap::new(),
            last_used: HashMap::new(),
            max_cache_size: 15, // Increased cache size for better performance
            total_memory_estimate: 0,
            max_memory_mb: 512, // 512MB texture cache limit
        }
    }

    fn get_or_create_optimized<F>(&mut self, key: String, ctx: &egui::Context, create_fn: F) -> &egui::TextureHandle
    where
        F: FnOnce() -> DynamicImage,
    {
        self.last_used.insert(key.clone(), Instant::now());

        if !self.images.contains_key(&key) {
            // Clean cache if needed
            if self.images.len() >= self.max_cache_size || self.total_memory_estimate > self.max_memory_mb * 1024 * 1024 {
                self.cleanup_cache();
            }

            let image = create_fn();
            let memory_estimate = (image.width() * image.height() * 4) as usize; // RGBA estimation
            self.total_memory_estimate += memory_estimate;

            let optimized_image = OptimizedImage::new(image);
            self.images.insert(key.clone(), optimized_image);
        }

        // Get the texture from the optimized image
        let optimized_image = self.images.get_mut(&key).unwrap();
        optimized_image.create_optimized_texture(ctx, key.clone())
    }

    fn cleanup_cache(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(45); // Keep images for 45 seconds

        let mut to_remove = Vec::new();
        for (key, &last_used) in &self.last_used {
            if last_used < cutoff {
                to_remove.push(key.clone());
            }
        }

        // If memory pressure is high, remove more aggressively
        if self.total_memory_estimate > self.max_memory_mb * 1024 * 1024 {
            let mut entries: Vec<_> = self.last_used.iter().collect();
            entries.sort_by_key(|(_, &time)| time);

            let additional_to_remove = (self.images.len() / 2).max(3); // Remove at least half when under memory pressure
            for (key, _) in entries.iter().take(additional_to_remove) {
                to_remove.push((*key).clone());
            }
        }

        // Remove selected items and update memory estimate
        for key in to_remove {
            if let Some(optimized_image) = self.images.remove(&key) {
                let memory_estimate = (optimized_image.original.width() * optimized_image.original.height() * 4) as usize;
                self.total_memory_estimate = self.total_memory_estimate.saturating_sub(memory_estimate);
            }
            self.last_used.remove(&key);
        }
    }

    fn clear(&mut self) {
        self.images.clear();
        self.last_used.clear();
        self.total_memory_estimate = 0;
    }

    fn get_memory_usage_mb(&self) -> f32 {
        self.total_memory_estimate as f32 / (1024.0 * 1024.0)
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct SimpleNeuralNetwork {
    weights1: Array2<f32>, // 1000 -> 128
    bias1: Array1<f32>,
    weights2: Array2<f32>, // 128 -> 64
    bias2: Array1<f32>,
    weights3: Array2<f32>, // 64 -> 5
    bias3: Array1<f32>,
    learning_rate: f32,
    // Adaptive learning parameters
    recent_accuracy: f32,
    training_count: usize,
    last_loss: f32,
}

#[cfg(feature = "gpu")]
#[derive(Clone)]
struct GpuNeuralNetwork {
    device: CandleDevice,
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
    varmap: VarMap,
    learning_rate: f32,
    recent_accuracy: f32,
    training_count: usize,
    last_loss: f32,
}

impl SimpleNeuralNetwork {
    fn new() -> Self {
        // Xavier initialization using ndarray-rand - Updated for 1000 input features
        let weights1 = Array2::random((1000, 128), Uniform::new(-0.1, 0.1));
        let bias1 = Array1::zeros(128);
        let weights2 = Array2::random((128, 64), Uniform::new(-0.1, 0.1));
        let bias2 = Array1::zeros(64);
        let weights3 = Array2::random((64, 5), Uniform::new(-0.1, 0.1));
        let bias3 = Array1::zeros(5);

        SimpleNeuralNetwork {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
            learning_rate: 0.001,
            recent_accuracy: 0.0,
            training_count: 0,
            last_loss: 0.0,
        }
    }

    fn relu(x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| v.max(0.0))
    }

    fn softmax(x: &Array1<f32>) -> Array1<f32> {
        let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum = exp_x.sum();
        exp_x / sum
    }

    fn forward(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Layer 1
        let z1 = self.weights1.t().dot(input) + &self.bias1;
        let a1 = Self::relu(&z1);

        // Layer 2
        let z2 = self.weights2.t().dot(&a1) + &self.bias2;
        let a2 = Self::relu(&z2);

        // Layer 3 (output)
        let z3 = self.weights3.t().dot(&a2) + &self.bias3;
        let output = Self::softmax(&z3);

        (a1, a2, output)
    }

    fn predict(&self, features: &[f32]) -> Result<u8> {
        if features.len() != 1000 {
            return Err(anyhow::anyhow!("Expected 1000 features, got {}", features.len()));
        }

        let input = Array1::from_vec(features.to_vec());
        let (_, _, output) = self.forward(&input);

        let predicted_class = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(predicted_class as u8 + 1) // Convert 0-4 to 1-5
    }

    fn train(&mut self, training_data: &[ImageRating]) -> Result<()> {
        if training_data.len() < 10 {
            return Ok(()); // Need minimum data for training
        }

        // Adaptive learning: adjust based on recent performance
        self.adapt_learning_rate(training_data)?;

        // Use incremental training for better continuous learning
        if self.training_count > 0 && training_data.len() > 100 {
            // For established models, do incremental training on recent data
            let recent_data_size = (training_data.len() / 4).max(50).min(200);
            let recent_data = &training_data[training_data.len() - recent_data_size..];
            self.incremental_train(recent_data)?;
        } else {
            // Full training for new models or small datasets
            self.full_train(training_data)?;
        }

        self.training_count += 1;
        Ok(())
    }

    fn adapt_learning_rate(&mut self, training_data: &[ImageRating]) -> Result<()> {
        // Calculate recent accuracy on a validation subset
        if training_data.len() > 50 {
            let validation_size = (training_data.len() / 10).max(20).min(100);
            let validation_data = &training_data[training_data.len() - validation_size..];

            let mut correct_predictions = 0;
            let mut total_loss = 0.0;

            for sample in validation_data {
                if sample.features.len() == 1000 {
                    if let Ok(prediction) = self.predict(&sample.features) {
                        if prediction == sample.rating {
                            correct_predictions += 1;
                        }
                    }

                    // Calculate loss for this sample
                    let input = Array1::from_vec(sample.features.clone());
                    let (_, _, output) = self.forward(&input);
                    let target_class = (sample.rating - 1) as usize;

                    // Cross-entropy loss
                    let predicted_prob = output[target_class].max(1e-7); // Avoid log(0)
                    total_loss += -predicted_prob.ln();
                }
            }

            let current_accuracy = correct_predictions as f32 / validation_data.len() as f32;
            let current_loss = total_loss / validation_data.len() as f32;

            // Adaptive learning rate adjustment
            if self.training_count > 0 {
                if current_accuracy > self.recent_accuracy {
                    // Improving: slightly increase learning rate
                    self.learning_rate = (self.learning_rate * 1.05).min(0.01);
                } else if current_loss > self.last_loss {
                    // Getting worse: decrease learning rate
                    self.learning_rate = (self.learning_rate * 0.9).max(0.0001);
                }
            }

            self.recent_accuracy = current_accuracy;
            self.last_loss = current_loss;
        }

        Ok(())
    }

    fn incremental_train(&mut self, recent_data: &[ImageRating]) -> Result<()> {
        // Lighter training for continuous improvement
        let epochs = 10; // Fewer epochs for incremental training
        let batch_size = recent_data.len().min(16); // Smaller batches

        for _ in 0..epochs {
            let mut shuffled_data = recent_data.to_vec();
            shuffled_data.shuffle(&mut rng());

            for batch in shuffled_data.chunks(batch_size) {
                self.train_batch(batch)?;
            }
        }

        Ok(())
    }

    fn full_train(&mut self, training_data: &[ImageRating]) -> Result<()> {
        // Full training for new models or major updates
        let epochs = if self.training_count == 0 { 100 } else { 50 }; // More epochs for initial training
        let batch_size = training_data.len().min(32);

        for _ in 0..epochs {
            let mut shuffled_data = training_data.to_vec();
            shuffled_data.shuffle(&mut rng());

            for batch in shuffled_data.chunks(batch_size) {
                self.train_batch(batch)?;
            }
        }

        Ok(())
    }

    fn train_batch(&mut self, batch: &[ImageRating]) -> Result<()> {
        let batch_size = batch.len() as f32;

        // Initialize gradients with explicit types
        let mut grad_w1: Array2<f32> = Array2::zeros(self.weights1.dim());
        let mut grad_b1: Array1<f32> = Array1::zeros(self.bias1.dim());
        let mut grad_w2: Array2<f32> = Array2::zeros(self.weights2.dim());
        let mut grad_b2: Array1<f32> = Array1::zeros(self.bias2.dim());
        let mut grad_w3: Array2<f32> = Array2::zeros(self.weights3.dim());
        let mut grad_b3: Array1<f32> = Array1::zeros(self.bias3.dim());

        for sample in batch {
            if sample.features.len() != 1000 {
                continue;
            }

            let input = Array1::from_vec(sample.features.clone());
            let target_class = (sample.rating - 1) as usize; // Convert 1-5 to 0-4

            // Forward pass
            let (a1, a2, output) = self.forward(&input);

            // Create one-hot target
            let mut target = Array1::zeros(5);
            target[target_class] = 1.0;

            // Backward pass
            let delta3 = &output - &target;
            grad_w3 = grad_w3 + a2.insert_axis(ndarray::Axis(1)).dot(&delta3.clone().insert_axis(ndarray::Axis(0)));
            grad_b3 = grad_b3 + &delta3;

            let delta2 = self.weights3.dot(&delta3);
            let delta2_relu = delta2.mapv(|v| if v > 0.0 { v } else { 0.0 });
            grad_w2 = grad_w2 + a1.insert_axis(ndarray::Axis(1)).dot(&delta2_relu.clone().insert_axis(ndarray::Axis(0)));
            grad_b2 = grad_b2 + &delta2_relu;

            let delta1 = self.weights2.dot(&delta2_relu);
            let delta1_relu = delta1.mapv(|v| if v > 0.0 { v } else { 0.0 });
            grad_w1 = grad_w1 + input.insert_axis(ndarray::Axis(1)).dot(&delta1_relu.clone().insert_axis(ndarray::Axis(0)));
            grad_b1 = grad_b1 + &delta1_relu;
        }

        // Update weights
        self.weights1 = &self.weights1 - &(grad_w1 * (self.learning_rate / batch_size));
        self.bias1 = &self.bias1 - &(grad_b1 * (self.learning_rate / batch_size));
        self.weights2 = &self.weights2 - &(grad_w2 * (self.learning_rate / batch_size));
        self.bias2 = &self.bias2 - &(grad_b2 * (self.learning_rate / batch_size));
        self.weights3 = &self.weights3 - &(grad_w3 * (self.learning_rate / batch_size));
        self.bias3 = &self.bias3 - &(grad_b3 * (self.learning_rate / batch_size));

        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl GpuNeuralNetwork {
    fn new(device: CandleDevice) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let layer1 = candle_nn::linear(1000, 128, vs.pp("layer1"))?;
        let layer2 = candle_nn::linear(128, 64, vs.pp("layer2"))?;
        let layer3 = candle_nn::linear(64, 5, vs.pp("layer3"))?;

        Ok(GpuNeuralNetwork {
            device,
            layer1,
            layer2,
            layer3,
            varmap,
            learning_rate: 0.001,
            recent_accuracy: 0.0,
            training_count: 0,
            last_loss: 0.0,
        })
    }

    fn forward(&self, input: &CandleTensor) -> Result<CandleTensor> {
        let x = self.layer1.forward(input)?;
        let x = x.relu()?;
        let x = self.layer2.forward(&x)?;
        let x = x.relu()?;
        let x = self.layer3.forward(&x)?;
        let output = candle_nn::ops::softmax(&x, 1)?;
        Ok(output)
    }

    fn predict(&self, features: &[f32]) -> Result<u8> {
        if features.len() != 1000 {
            return Err(anyhow::anyhow!("Expected 1000 features, got {}", features.len()));
        }

        let input = CandleTensor::from_slice(features, (1, 1000), &self.device)?;
        let output = self.forward(&input)?;
        let output_vec = output.to_vec2::<f32>()?;

        let predicted_class = output_vec[0].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(predicted_class as u8 + 1)
    }

    fn train(&mut self, training_data: &[ImageRating]) -> Result<()> {
        if training_data.len() < 10 {
            return Ok(());
        }

        // Adaptive learning
        self.adapt_learning_rate(training_data)?;

        // Use incremental training for better continuous learning
        if self.training_count > 0 && training_data.len() > 100 {
            let recent_data_size = (training_data.len() / 4).max(50).min(200);
            let recent_data = &training_data[training_data.len() - recent_data_size..];
            self.incremental_train(recent_data)?;
        } else {
            self.full_train(training_data)?;
        }

        self.training_count += 1;
        Ok(())
    }

    fn adapt_learning_rate(&mut self, training_data: &[ImageRating]) -> Result<()> {
        if training_data.len() > 50 {
            let validation_size = (training_data.len() / 10).max(20).min(100);
            let validation_data = &training_data[training_data.len() - validation_size..];

            let mut correct_predictions = 0;
            let mut total_loss = 0.0;

            for sample in validation_data {
                if sample.features.len() == 1000 {
                    if let Ok(prediction) = self.predict(&sample.features) {
                        if prediction == sample.rating {
                            correct_predictions += 1;
                        }
                    }

                    // Calculate loss
                    let input = CandleTensor::from_slice(&sample.features, (1, 1000), &self.device)?;
                    let output = self.forward(&input)?;
                    let target_class = (sample.rating - 1) as usize;

                    let output_vec = output.to_vec2::<f32>()?;
                    let predicted_prob = output_vec[0][target_class].max(1e-7);
                    total_loss += -predicted_prob.ln();
                }
            }

            let current_accuracy = correct_predictions as f32 / validation_data.len() as f32;
            let current_loss = total_loss / validation_data.len() as f32;

            if self.training_count > 0 {
                if current_accuracy > self.recent_accuracy {
                    self.learning_rate = (self.learning_rate * 1.05).min(0.01);
                } else if current_loss > self.last_loss {
                    self.learning_rate = (self.learning_rate * 0.9).max(0.0001);
                }
            }

            self.recent_accuracy = current_accuracy;
            self.last_loss = current_loss;
        }

        Ok(())
    }

    fn incremental_train(&mut self, recent_data: &[ImageRating]) -> Result<()> {
        self.train_epochs(recent_data, 10)
    }

    fn full_train(&mut self, training_data: &[ImageRating]) -> Result<()> {
        let epochs = if self.training_count == 0 { 100 } else { 50 };
        self.train_epochs(training_data, epochs)
    }

    fn train_epochs(&mut self, training_data: &[ImageRating], epochs: usize) -> Result<()> {
        let params = ParamsAdamW {
            lr: self.learning_rate as f64,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.varmap.all_vars(), params)?;

        for _ in 0..epochs {
            let mut shuffled_data = training_data.to_vec();
            shuffled_data.shuffle(&mut rng());

            let batch_size = training_data.len().min(32);
            for batch in shuffled_data.chunks(batch_size) {
                self.train_batch(batch, &mut optimizer)?;
            }
        }

        Ok(())
    }

    fn train_batch(&self, batch: &[ImageRating], optimizer: &mut AdamW) -> Result<()> {
        let batch_size = batch.len();
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for sample in batch {
            if sample.features.len() == 1000 {
                inputs.extend_from_slice(&sample.features);
                targets.push((sample.rating - 1) as usize);
            }
        }

        if inputs.is_empty() {
            return Ok(());
        }

        let input_tensor = CandleTensor::from_slice(&inputs, (batch_size, 1000), &self.device)?;
        let output = self.forward(&input_tensor)?;

        // Create target tensor with class indices (not one-hot)
        let target_indices: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        let target_tensor = CandleTensor::from_slice(&target_indices, batch_size, &self.device)?;

        // Cross-entropy loss expects class indices, not one-hot encoding
        let loss = candle_nn::loss::cross_entropy(&output, &target_tensor)?;

        optimizer.backward_step(&loss)?;

        Ok(())
    }
}

#[derive(Clone)]
struct RatingPredictor {
    #[cfg(feature = "gpu")]
    gpu_network: Option<GpuNeuralNetwork>,
    cpu_network: SimpleNeuralNetwork,
    model_file: PathBuf,
    #[allow(dead_code)]
    use_gpu: bool,
}

impl RatingPredictor {
    fn new(model_file: PathBuf) -> Result<Self> {
        let cpu_network = if model_file.exists() {
            // Try to load existing model
            match fs::read_to_string(&model_file) {
                Ok(data) => serde_json::from_str(&data).unwrap_or_else(|_| SimpleNeuralNetwork::new()),
                Err(_) => SimpleNeuralNetwork::new(),
            }
        } else {
            SimpleNeuralNetwork::new()
        };

        #[cfg(feature = "gpu")]
        let (gpu_network, use_gpu) = {
            // Try to initialize GPU
            #[cfg(all(target_os = "windows", target_env = "gnu"))]
            {
                // Windows GNU: CUDA is not supported due to linking issues
                if !GPU_MESSAGE_PRINTED.swap(true, Ordering::Relaxed) {
                    println!("Windows GNU detected: CUDA not supported, using CPU-only mode");
                    println!("For GPU acceleration on Windows, please use the MSVC toolchain:");
                    println!("  rustup default stable-x86_64-pc-windows-msvc");
                    println!("  cargo build --features cuda --target x86_64-pc-windows-msvc");
                }
                (None, false)
            }
            #[cfg(not(all(target_os = "windows", target_env = "gnu")))]
            {
                match CandleDevice::new_cuda(0) {
                    Ok(device) => {
                        match GpuNeuralNetwork::new(device) {
                            Ok(gpu_net) => {
                                if !GPU_MESSAGE_PRINTED.swap(true, Ordering::Relaxed) {
                                    println!("GPU acceleration enabled with CUDA");
                                }
                                (Some(gpu_net), true)
                            }
                            Err(e) => {
                                if !GPU_MESSAGE_PRINTED.swap(true, Ordering::Relaxed) {
                                    eprintln!("Failed to initialize GPU network: {}, falling back to CPU", e);
                                }
                                (None, false)
                            }
                        }
                    }
                    Err(_) => {
                        // Try Metal for Apple Silicon
                        match CandleDevice::new_metal(0) {
                            Ok(device) => {
                                match GpuNeuralNetwork::new(device) {
                                    Ok(gpu_net) => {
                                        if !GPU_MESSAGE_PRINTED.swap(true, Ordering::Relaxed) {
                                            println!("GPU acceleration enabled with Metal");
                                        }
                                        (Some(gpu_net), true)
                                    }
                                    Err(e) => {
                                        if !GPU_MESSAGE_PRINTED.swap(true, Ordering::Relaxed) {
                                            eprintln!("Failed to initialize GPU network: {}, falling back to CPU", e);
                                        }
                                        (None, false)
                                    }
                                }
                            }
                            Err(_) => {
                                if !GPU_MESSAGE_PRINTED.swap(true, Ordering::Relaxed) {
                                    println!("No GPU available, using CPU");
                                }
                                (None, false)
                            }
                        }
                    }
                }
            }
        };

        #[cfg(not(feature = "gpu"))]
        let use_gpu = false;

        Ok(RatingPredictor {
            #[cfg(feature = "gpu")]
            gpu_network,
            cpu_network,
            model_file,
            #[allow(dead_code)]
            use_gpu,
        })
    }

    fn predict(&self, features: &[f32]) -> Result<u8> {
        #[cfg(feature = "gpu")]
        if self.use_gpu {
            if let Some(ref gpu_net) = self.gpu_network {
                return gpu_net.predict(features);
            }
        }

        self.cpu_network.predict(features)
    }

    fn train(&mut self, training_data: &[ImageRating]) -> Result<()> {
        #[cfg(feature = "gpu")]
        if self.use_gpu {
            if let Some(ref mut gpu_net) = self.gpu_network {
                gpu_net.train(training_data)?;
                // Also train CPU network for serialization
                self.cpu_network.train(training_data)?;
            }
        } else {
            self.cpu_network.train(training_data)?;
        }

        #[cfg(not(feature = "gpu"))]
        self.cpu_network.train(training_data)?;

        // Save the trained CPU model (GPU model can't be easily serialized)
        let model_json = serde_json::to_string_pretty(&self.cpu_network)?;
        fs::write(&self.model_file, model_json)?;

        Ok(())
    }

    fn get_performance_info(&self) -> String {
        #[cfg(feature = "gpu")]
        if self.use_gpu {
            return "GPU-accelerated".to_string();
        }

        "CPU-only".to_string()
    }
}

struct ImageViewer {
    images: Vec<PathBuf>,
    current_index: usize,
    loaded_images: HashMap<usize, DynamicImage>, // Changed to HashMap for better tracking
    texture_cache: TextureCache,
    favorites_dir: PathBuf,
    deleted_dir: PathBuf,
    favorite_counter: usize,
    flash_animation: FlashAnimation,
    state_file: PathBuf,
    current_directory: String,
    // Rating system fields
    predictor: Option<RatingPredictor>,
    shared_predictor: Arc<Mutex<Option<RatingPredictor>>>,
    ratings_data: Vec<ImageRating>,
    ratings_file: PathBuf,
    model_file: PathBuf,
    current_rating: Option<u8>,
    suggested_rating: Option<u8>,
    image_features_cache: HashMap<String, Vec<f32>>,
    // Background processing - simplified
    feature_extraction_rx: Receiver<(String, Vec<f32>)>,
    prediction_tx: Sender<(String, Vec<f32>)>,
    prediction_rx: Receiver<(String, u8)>,
    pending_suggestions: HashMap<String, bool>,
    // Simplified image processing - removed async loading for stability
    // Performance optimization fields
    last_repaint_request: Instant,
    dirty_state: bool,
    last_background_check: Instant,
    onnx_session: Arc<ort::Session>,
}

impl ImageViewer {
    fn new(directory: &Path) -> Result<Self> {
        let mut images = Vec::new();

        // Recursively find all image files
        for entry in WalkDir::new(directory)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp") {
                    images.push(path.to_path_buf());
                }
            }
        }

        if images.is_empty() {
            anyhow::bail!("No images found in the specified directory");
        }

        let (_tx, _rx) = channel::<(usize, PathBuf)>();
        let (_loading_tx, _loading_rx) = channel::<(usize, DynamicImage)>();

        // Background processing channels - simplified for predictions only
        let (_feature_tx, feature_rx) = channel::<(String, DynamicImage)>();
        let (feature_result_tx, feature_result_rx) = channel::<(String, Vec<f32>)>();
        let (prediction_tx, prediction_rx) = channel::<(String, Vec<f32>)>();
        let (prediction_result_tx, prediction_result_rx) = channel::<(String, u8)>();

        // Shared predictor for background thread
        let shared_predictor: Arc<Mutex<Option<RatingPredictor>>> = Arc::new(Mutex::new(None));
        let thread_predictor = shared_predictor.clone();

        // Start feature extraction thread
        let environment = Arc::new(Environment::builder().with_name("iv-onnx").build()?);
        let model_path = find_model_path()?;
        let session = Arc::new(SessionBuilder::new(&environment)?.with_model_from_file(model_path)?);
        let onnx_session_thread = session.clone();
        thread::spawn(move || {
            while let Ok((hash, image)) = feature_rx.recv() {
                let features = ImageViewer::extract_features_static(&image, &onnx_session_thread);
                let _ = feature_result_tx.send((hash, features));
            }
        });

        // Start prediction thread
        thread::spawn(move || {
            while let Ok((hash, features)) = prediction_rx.recv() {
                if let Ok(predictor_guard) = thread_predictor.lock() {
                    if let Some(ref predictor) = *predictor_guard {
                        match predictor.predict(&features) {
                            Ok(prediction) => {
                                let _ = prediction_result_tx.send((hash, prediction));
                            }
                            Err(e) => {
                                eprintln!("Background prediction failed: {}", e);
                            }
                        }
                    }
                }
            }
        });

        // Create favorites and deleted directories
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path.parent()
            .ok_or_else(|| anyhow::anyhow!("Could not get executable directory"))?;
        let favorites_dir = exe_dir.join("favorites");
        let deleted_dir = exe_dir.join("deleted");

        fs::create_dir_all(&favorites_dir)?;
        fs::create_dir_all(&deleted_dir)?;

        // Use global rating files (not directory-specific)
        let global_data_dir = if let Some(home_dir) = std::env::var_os("HOME") {
            PathBuf::from(home_dir).join(".iv_ratings")
        } else {
            exe_dir.join("global_ratings")
        };
        fs::create_dir_all(&global_data_dir)?;

        let state_file = exe_dir.join("app_state.json");
        let current_directory = directory.to_string_lossy().to_string();

        // Load last position if available
        let mut current_index = 0;
        if let Ok(state_data) = fs::read_to_string(&state_file) {
            if let Ok(state) = serde_json::from_str::<AppState>(&state_data) {
                if let Some(&last_pos) = state.last_positions.get(&current_directory) {
                    if last_pos < images.len() {
                        current_index = last_pos;
                    }
                }
            }
        }

        // Find the highest favorite number
        let favorite_counter = WalkDir::new(&favorites_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy();
                if name.starts_with("favorite-") {
                    name.split('.')
                        .next()
                        .and_then(|s| s.strip_prefix("favorite-"))
                        .and_then(|s| s.parse::<usize>().ok())
                } else {
                    None
                }
            })
            .max()
            .map(|n| n + 1)
            .unwrap_or(0);

        let mut viewer = ImageViewer {
            images,
            current_index,
            loaded_images: HashMap::new(),
            texture_cache: TextureCache::new(),
            favorites_dir,
            deleted_dir,
            favorite_counter,
            flash_animation: FlashAnimation::new(egui::Color32::WHITE), // Default flash color
            state_file,
            current_directory,
            // Rating system fields - now global
            predictor: None,
            shared_predictor,
            ratings_data: Vec::new(),
            ratings_file: global_data_dir.join("ratings.json"),
            model_file: global_data_dir.join("model.json"),
            current_rating: None,
            suggested_rating: None,
            image_features_cache: HashMap::new(),
            // Background processing - simplified
            feature_extraction_rx: feature_result_rx,
            prediction_tx: prediction_tx,
            prediction_rx: prediction_result_rx,
            pending_suggestions: HashMap::new(),
            // Simplified image processing - removed async loading for stability
            // Performance optimization fields
            last_repaint_request: Instant::now(),
            dirty_state: false,
            last_background_check: Instant::now(),
            onnx_session: session,
        };

        // Load initial images
        viewer.load_initial_images()?;

        // Load ratings data
        if let Err(e) = viewer.load_ratings() {
            eprintln!("Failed to load ratings: {}", e);
        }

        Ok(viewer)
    }

    fn load_initial_images(&mut self) -> Result<()> {
        // Load current image synchronously for immediate display
        self.load_image_sync(self.current_index);

        // Load a few nearby images for smooth navigation
        if self.current_index > 0 {
            self.load_image_sync(self.current_index - 1);
        }
        if self.current_index + 1 < self.images.len() {
            self.load_image_sync(self.current_index + 1);
        }

        Ok(())
    }

    fn load_image_sync(&mut self, index: usize) {
        if index < self.images.len() && !self.loaded_images.contains_key(&index) {
            let path = &self.images[index];
            match image::open(path) {
                Ok(img) => {
                    self.loaded_images.insert(index, img);
                }
                Err(e) => {
                    eprintln!("Failed to load image {}: {}", path.display(), e);
                }
            }
        }
    }

    fn load_image(&mut self, index: usize) -> Result<()> {
        self.load_image_sync(index);
        Ok(())
    }

    fn update_loaded_images(&mut self) {
        // Keep a reasonable cache of loaded images around current position
        let keep_range = 5; // Keep 5 images around current position

        // Remove images that are too far from current position
        let current = self.current_index;
        self.loaded_images.retain(|&index, _| {
            index >= current.saturating_sub(keep_range) &&
            index <= current.saturating_add(keep_range).min(self.images.len().saturating_sub(1))
        });

        // Load nearby images if not already loaded
        for offset in 0..=2 {
            if current >= offset {
                self.load_image_sync(current - offset);
            }
            if current + offset < self.images.len() {
                self.load_image_sync(current + offset);
            }
        }
    }

    fn next_image(&mut self) -> Result<()> {
        if self.current_index + 1 < self.images.len() {
            self.current_index += 1;
            // Only clear texture cache if no flash animation is active
            if !self.flash_animation.is_active() {
                self.texture_cache.clear();
            }
            // Load next image if needed
            if self.current_index + 2 < self.images.len() {
                self.load_image(self.current_index + 2)?;
            }
            // Load previous image if needed
            if self.current_index >= 2 {
                self.load_image(self.current_index - 2)?;
            }
            // Load current image
            self.load_image(self.current_index)?;

            // Update rating suggestion for new image
            self.update_suggestion();
        }
        Ok(())
    }

    fn previous_image(&mut self) -> Result<()> {
        if self.current_index > 0 {
            self.current_index -= 1;
            // Only clear texture cache if no flash animation is active
            if !self.flash_animation.is_active() {
                self.texture_cache.clear();
            }
            // Load next image if needed
            if self.current_index + 2 < self.images.len() {
                self.load_image(self.current_index + 2)?;
            }
            // Load previous image if needed
            if self.current_index >= 2 {
                self.load_image(self.current_index - 2)?;
            }
            // Load current image
            self.load_image(self.current_index)?;

            // Update rating suggestion for new image
            self.update_suggestion();
        }
        Ok(())
    }

    fn copy_to_favorites(&mut self) -> Result<()> {
        let current_image = &self.images[self.current_index];
        let extension = current_image
            .extension()
            .ok_or_else(|| anyhow::anyhow!("No file extension found"))?
            .to_string_lossy();

        let new_name = format!("favorite-{}.{}", self.favorite_counter, extension);
        let target_path = self.favorites_dir.join(new_name);

        fs::copy(current_image, &target_path)?;
        self.favorite_counter += 1;

        self.flash_animation = FlashAnimation::new(egui::Color32::WHITE);

        Ok(())
    }

    fn move_to_deleted(&mut self) -> Result<()> {
        if self.images.is_empty() {
            return Ok(());
        }

        let current_image = &self.images[self.current_index];
        let filename = current_image
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("No filename found"))?;

        let target_path = self.deleted_dir.join(filename);

        // Use copy + delete instead of rename to work across drives
        fs::copy(current_image, &target_path)?;
        fs::remove_file(current_image)?;

        // Remove the image from our list
        self.images.remove(self.current_index);

        // Adjust the loaded images
        self.loaded_images = HashMap::new();
        self.texture_cache.clear();

        // If we've deleted the last image, go to the previous one
        if self.current_index >= self.images.len() && !self.images.is_empty() {
            self.current_index = self.images.len() - 1;
        }

        // Set flash state to red
        self.flash_animation = FlashAnimation::new(egui::Color32::RED);

        // Reload images around the current index
        self.load_initial_images()?;

        Ok(())
    }

    fn save_state(&self) -> Result<()> {
        let mut state = if let Ok(state_data) = fs::read_to_string(&self.state_file) {
            serde_json::from_str::<AppState>(&state_data).unwrap_or_default()
        } else {
            AppState::default()
        };

        state.last_positions.insert(self.current_directory.clone(), self.current_index);

        let state_json = serde_json::to_string_pretty(&state)?;
        fs::write(&self.state_file, state_json)?;

        // Save detailed ratings data to global location
        let ratings_json = serde_json::to_string_pretty(&self.ratings_data)?;
        fs::write(&self.ratings_file, ratings_json)?;

        Ok(())
    }

    fn load_ratings(&mut self) -> Result<()> {
        if let Ok(ratings_data) = fs::read_to_string(&self.ratings_file) {
            if let Ok(ratings) = serde_json::from_str::<Vec<ImageRating>>(&ratings_data) {
                self.ratings_data = ratings;
            }
        }

        // Clear the feature cache since we've changed the feature extraction method
        self.image_features_cache.clear();

        // Remove the old model file since features have changed
        if self.model_file.exists() {
            let _ = fs::remove_file(&self.model_file);
        }

        // Initialize predictor if we have enough data
        if self.ratings_data.len() >= 500 {
            match RatingPredictor::new(self.model_file.clone()) {
                Ok(mut predictor) => {
                    if let Err(e) = predictor.train(&self.ratings_data) {
                        eprintln!("Failed to train predictor: {}", e);
                    } else {
                        // Update both local and shared predictors
                        if let Ok(mut shared) = self.shared_predictor.lock() {
                            *shared = Some(predictor.clone());
                        }
                        self.predictor = Some(predictor);
                    }
                }
                Err(e) => eprintln!("Failed to create predictor: {}", e),
            }
        }

        Ok(())
    }

    fn compute_image_hash(&self, image: &DynamicImage) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&image.to_rgba8().into_raw());
        hex::encode(hasher.finalize())
    }

    fn extract_features_static(image: &DynamicImage, session: &ort::Session) -> Vec<f32> {
        // Resize and normalize image to 224x224, RGB
        let resized = image.resize_exact(224, 224, image::imageops::FilterType::Lanczos3).to_rgb8();
        let mut input = Array4::<f32>::zeros((1, 3, 224, 224));
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        for y in 0..224 {
            for x in 0..224 {
                let pixel = resized.get_pixel(x, y);
                for c in 0..3 {
                    input[[0, c, y as usize, x as usize]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                }
            }
        }
        // Run through ONNX model
        let input_dyn = input.into_dyn();
        let input_cow = CowArray::from(&input_dyn);
        let input_tensor = Value::from_array(session.allocator(), &input_cow).expect("Failed to create input tensor");
        let outputs = session.run(vec![input_tensor]).expect("ONNX inference failed");
        let features_tensor = outputs[0].try_extract::<f32>().expect("Failed to extract output");
        let features: Vec<f32> = features_tensor.view().as_slice().unwrap().to_vec();
        features
    }

    fn rate_current_image(&mut self, rating: u8) -> Result<()> {
        if let Some(image) = self.loaded_images.get(&self.current_index) {
            let hash = self.compute_image_hash(image);
            let features = if let Some(cached) = self.image_features_cache.get(&hash) {
                cached.clone()
            } else {
                let features = Self::extract_features_static(image, &self.onnx_session);
                self.image_features_cache.insert(hash.clone(), features.clone());
                features
            };

            // Update or add rating
            if let Some(existing) = self.ratings_data.iter_mut().find(|r| r.image_hash == hash) {
                existing.rating = rating;
            } else {
                self.ratings_data.push(ImageRating {
                    image_hash: hash,
                    rating,
                    features,
                });
            }

            self.current_rating = Some(rating);

            // Retrain model if we have enough data
            if self.ratings_data.len() >= 500 {
                if self.predictor.is_none() {
                    match RatingPredictor::new(self.model_file.clone()) {
                        Ok(predictor) => {
                            // Update both local and shared predictors
                            if let Ok(mut shared) = self.shared_predictor.lock() {
                                *shared = Some(predictor.clone());
                            }
                            self.predictor = Some(predictor);
                        }
                        Err(e) => eprintln!("Failed to create predictor: {}", e),
                    }
                }

                if let Some(ref mut predictor) = self.predictor {
                    if let Err(e) = predictor.train(&self.ratings_data) {
                        eprintln!("Failed to retrain model: {}", e);
                    } else {
                        // Update shared predictor after retraining
                        if let Ok(mut shared) = self.shared_predictor.lock() {
                            *shared = Some(predictor.clone());
                        }
                    }
                }
            }

            // Flash animation will be set after advancing to next image
        }

        Ok(())
    }

    fn update_suggestion(&mut self) {
        // Process any completed background work first
        self.process_background_results();

        if let Some(image) = self.loaded_images.get(&self.current_index) {
            let hash = self.compute_image_hash(image);

            // Check if we already have a rating for this image
            if let Some(existing) = self.ratings_data.iter().find(|r| r.image_hash == hash) {
                self.current_rating = Some(existing.rating);
                self.suggested_rating = None;
                return;
            }

            // Check if we have cached features
            if let Some(cached_features) = self.image_features_cache.get(&hash) {
                // We have features, check if we can predict
                if let Some(predictor) = &self.predictor {
                    match predictor.predict(cached_features) {
                        Ok(prediction) => {
                            self.suggested_rating = Some(prediction);
                            self.current_rating = None;
                        }
                        Err(e) => {
                            eprintln!("Failed to predict rating: {}", e);
                            self.suggested_rating = None;
                            self.current_rating = None;
                        }
                    }
                } else {
                    self.suggested_rating = None;
                    self.current_rating = None;
                }
            } else {
                // Extract features synchronously for immediate use
                let features = Self::extract_features_static(image, &self.onnx_session);
                self.image_features_cache.insert(hash.clone(), features.clone());

                // Predict immediately if we have a predictor
                if let Some(predictor) = &self.predictor {
                    match predictor.predict(&features) {
                        Ok(prediction) => {
                            self.suggested_rating = Some(prediction);
                            self.current_rating = None;
                        }
                        Err(e) => {
                            eprintln!("Failed to predict rating: {}", e);
                            self.suggested_rating = None;
                            self.current_rating = None;
                        }
                    }
                } else {
                    self.suggested_rating = None;
                    self.current_rating = None;
                }
            }
        }
    }

    fn process_background_results(&mut self) {
        // Simplified - only process prediction results from background thread

        // Process completed feature extractions (legacy support)
        while let Ok((hash, features)) = self.feature_extraction_rx.try_recv() {
            self.image_features_cache.insert(hash.clone(), features.clone());
            self.pending_suggestions.remove(&hash);

            // If we have a predictor, start prediction in background
            if self.predictor.is_some() {
                if let Err(e) = self.prediction_tx.send((hash, features)) {
                    eprintln!("Failed to send features for prediction: {}", e);
                }
            }
        }

        // Process completed predictions
        while let Ok((hash, prediction)) = self.prediction_rx.try_recv() {
            // Check if this prediction is for the current image
            if let Some(current_image) = self.loaded_images.get(&self.current_index) {
                let current_hash = self.compute_image_hash(current_image);
                if hash == current_hash {
                    // Check if we don't already have a rating for this image
                    if !self.ratings_data.iter().any(|r| r.image_hash == hash) {
                        self.suggested_rating = Some(prediction);
                        self.current_rating = None;
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    fn preextract_nearby_features(&mut self) {
        // Simplified - extract features for loaded images that don't have them yet
        for (&_index, image) in &self.loaded_images {
            let hash = self.compute_image_hash(image);
            if !self.image_features_cache.contains_key(&hash) &&
               !self.pending_suggestions.contains_key(&hash) {
                // Extract features synchronously for better reliability
                let features = Self::extract_features_static(image, &self.onnx_session);
                self.image_features_cache.insert(hash, features);
            }
        }
    }

    #[allow(dead_code)]
    fn should_defer_heavy_operations(&self) -> bool {
        // Defer heavy operations if we're processing too frequently (likely window movement)
        let now = Instant::now();
        let _should_repaint = now.duration_since(self.last_repaint_request) > Duration::from_millis(16); // ~60 FPS max
        false
    }
}

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Optimize repaint frequency and reduce work when window is being dragged
        let now = Instant::now();
        let _should_repaint = now.duration_since(self.last_repaint_request) > Duration::from_millis(16); // ~60 FPS max

        // Detect if window is being interacted with (moved/resized) by checking if we're getting rapid updates
        let is_window_interaction = now.duration_since(self.last_repaint_request) < Duration::from_millis(8);

        // Reduce background processing during window interactions
        let should_process_background = !is_window_interaction &&
            now.duration_since(self.last_background_check) > Duration::from_millis(33); // 30fps for background work

        // Update window title only when needed
        if !is_window_interaction {
            if let Some(current_path) = self.images.get(self.current_index) {
                if let Some(filename) = current_path.file_name() {
                    if let Some(filename_str) = filename.to_str() {
                        ctx.send_viewport_cmd(ViewportCommand::Title(format!("{} | IV", filename_str)));
                    }
                }
            }
        }

        // Only update loaded images if not dragging window
        if !is_window_interaction {
            self.update_loaded_images();
        }

        // Update flash animation - always update when active
        let flash_intensity = if self.flash_animation.is_active() {
            let intensity = self.flash_animation.update();
            // Always repaint during flash animations for visibility
            if intensity > 0.0 {
                ctx.request_repaint();
                self.dirty_state = true;
            } else {
                // Flash just finished, clean up texture cache
                self.texture_cache.clear();
            }
            intensity
        } else {
            0.0
        };

        // Update rating suggestions when images are loaded - throttle during window movement
        if should_process_background {
            self.update_suggestion();
            self.last_background_check = now;
        }

        // Process background results less frequently during window interactions
        if should_process_background {
            self.process_background_results();
        }

        // Handle auto-advance after rating (wait for flash to be visible)
        // Commented out auto-advance logic - now advance immediately during rating
        // if let Some(advance_time) = self.auto_advance_time {
        //     if now.duration_since(advance_time) > Duration::from_millis(250) { // Wait 250ms to see flash
        //         let _ = self.next_image();
        //         self.auto_advance_time = None;
        //         needs_immediate_repaint = true;
        //     }
        // }

        // Handle keyboard input
        let mut needs_immediate_repaint = false;

        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            let _ = self.next_image();
            needs_immediate_repaint = true;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            let _ = self.previous_image();
            needs_immediate_repaint = true;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            ctx.send_viewport_cmd(ViewportCommand::Close);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Plus)) {
            if let Err(e) = self.copy_to_favorites() {
                eprintln!("Failed to copy image to favorites: {}", e);
            }
            needs_immediate_repaint = true;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            if let Err(e) = self.move_to_deleted() {
                eprintln!("Failed to move image to deleted folder: {}", e);
            }
            needs_immediate_repaint = true;
        }

        // Handle rating input (number keys 1-5)
        for (key, rating) in [
            (egui::Key::Num1, 1),
            (egui::Key::Num2, 2),
            (egui::Key::Num3, 3),
            (egui::Key::Num4, 4),
            (egui::Key::Num5, 5),
        ] {
            if ctx.input(|i| i.key_pressed(key)) {
                if let Err(e) = self.rate_current_image(rating) {
                    eprintln!("Failed to rate image: {}", e);
                } else {
                    // Advance to next image first
                    let _ = self.next_image();

                    // Then set flash animation to play on the advanced image
                    self.flash_animation = FlashAnimation::new(match rating {
                        1 => egui::Color32::from_rgb(255, 100, 100), // Red
                        2 => egui::Color32::from_rgb(255, 165, 0),   // Orange
                        3 => egui::Color32::from_rgb(255, 255, 100), // Yellow
                        4 => egui::Color32::from_rgb(144, 238, 144), // Light green
                        5 => egui::Color32::from_rgb(100, 255, 100), // Green
                        _ => egui::Color32::WHITE,
                    });

                    ctx.request_repaint();
                    needs_immediate_repaint = true;
                }
            }
        }

        // Debug feature extraction (press D key)
        if ctx.input(|i| i.key_pressed(egui::Key::D)) {
            if let Some(image) = self.loaded_images.get(&self.current_index) {
                let features = Self::extract_features_static(image, &self.onnx_session);
                println!("\n=== DEBUG: Feature Analysis ===");
                println!("Image size: {}x{}", image.width(), image.height());
                println!("Feature vector size: {} (ResNet-50 output)", features.len());

                // Show first few features for debugging
                if features.len() >= 10 {
                    println!("First 10 features: {:?}", &features[0..10]);
                }
                if features.len() >= 100 {
                    println!("Features 90-100: {:?}", &features[90..100]);
                }

                if let Some(predictor) = &self.predictor {
                    match predictor.predict(&features) {
                        Ok(prediction) => {
                            println!("\nAI prediction: {}", prediction);
                            println!("Using: {}", predictor.get_performance_info());
                        }
                        Err(e) => println!("Prediction error: {}", e),
                    }
                } else {
                    println!("AI not yet trained (need {} more ratings)", 500 - self.ratings_data.len().min(500));
                }
                println!("===============================\n");
            }
        }

        // Test simplified processing (press T key)
        if ctx.input(|i| i.key_pressed(egui::Key::T)) {
            println!("Image processing is now synchronous for better stability:");
            println!("- Images load immediately when navigating");
            println!("- Features extract synchronously for instant AI predictions");
            println!("- Texture cache manages memory efficiently");
            println!("- No more race conditions or out-of-order issues");
        }

        // Handle mouse input
        if ctx.input(|i| i.pointer.primary_clicked()) {
            let _ = self.next_image();
            needs_immediate_repaint = true;
        }
        if ctx.input(|i| i.pointer.secondary_clicked()) {
            let _ = self.previous_image();
            needs_immediate_repaint = true;
        }

        // Only request repaint when necessary - always repaint during flash animations
        if needs_immediate_repaint || self.flash_animation.is_active() {
            self.dirty_state = true;
        }

        // Always request repaint if we have dirty state (removed window interaction check for flash)
        if self.dirty_state {
            ctx.request_repaint();
            self.last_repaint_request = now;
            self.dirty_state = false;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Rating UI at the top
            ui.horizontal(|ui| {
                ui.label("Rating:");

                // Show current rating or suggestion
                if let Some(rating) = self.current_rating {
                    ui.colored_label(egui::Color32::GREEN, format!("★ {}", rating));
                } else if let Some(suggestion) = self.suggested_rating {
                    ui.colored_label(egui::Color32::YELLOW, format!("AI suggests: {}", suggestion));
                    ui.label("(Press 1-5 to rate)");
                } else {
                    ui.label("Press 1-5 to rate this image");
                }

                ui.separator();

                // Show training progress
                let rated_count = self.ratings_data.len();
                if rated_count < 500 {
                    ui.label(format!("Rated: {}/500 ({}% to AI suggestions)",
                        rated_count,
                        (rated_count * 100) / 500));
                } else {
                    ui.colored_label(egui::Color32::GREEN,
                        format!("AI Active! ({} images rated)", rated_count));
                }

                ui.separator();

                // Show GPU acceleration status and performance info
                if let Some(ref predictor) = self.predictor {
                    let performance_info = predictor.get_performance_info();
                    let color = if performance_info.contains("GPU") {
                        egui::Color32::from_rgb(100, 255, 100) // Green for GPU
                    } else {
                        egui::Color32::from_rgb(255, 255, 100) // Yellow for CPU
                    };
                    ui.colored_label(color, performance_info);
                }

                // Show texture cache usage
                let memory_usage = self.texture_cache.get_memory_usage_mb();
                let memory_color = if memory_usage > 400.0 {
                    egui::Color32::from_rgb(255, 100, 100) // Red for high usage
                } else if memory_usage > 200.0 {
                    egui::Color32::from_rgb(255, 255, 100) // Yellow for medium usage
                } else {
                    egui::Color32::from_rgb(100, 255, 100) // Green for low usage
                };
                ui.colored_label(memory_color, format!("Cache: {:.1}MB", memory_usage));
            });

            ui.separator();

            if let Some(img) = self.loaded_images.get(&self.current_index) {
                let size = [img.width() as f32, img.height() as f32];

                // Use optimized texture cache with better key
                let texture_key = format!("image_{}_{}", self.current_index, img.width() * img.height());
                let texture = self.texture_cache.get_or_create_optimized(texture_key, ctx, || {
                    img.clone()
                });

                let available_size = ui.available_size();
                let scale = (available_size.x / size[0]).min((available_size.y - 40.0) / size[1]); // Reserve space for progress bar
                let scaled_size = egui::vec2(size[0] * scale, size[1] * scale);

                let response = ui.centered_and_justified(|ui| {
                    let mut image_widget = egui::Image::new((texture.id(), scaled_size))
                        .sense(egui::Sense::click_and_drag());

                    // Apply flash effect with much stronger visibility
                    if flash_intensity > 0.0 {
                        let flash_tint = match self.flash_animation.color {
                            egui::Color32::WHITE => egui::Color32::from_rgba_premultiplied(255, 255, 255, (flash_intensity * 200.0) as u8),
                            egui::Color32::RED => egui::Color32::from_rgba_premultiplied(255, 100, 100, (flash_intensity * 180.0) as u8),
                            color => {
                                let [r, g, b, _] = color.to_array();
                                egui::Color32::from_rgba_premultiplied(r, g, b, (flash_intensity * 200.0) as u8)
                            }
                        };
                        image_widget = image_widget.tint(flash_tint);
                    }

                    ui.add(image_widget)
                });

                if response.inner.clicked() {
                    let _ = self.next_image();
                    ctx.request_repaint();
                }
                if response.inner.secondary_clicked() {
                    let _ = self.previous_image();
                    ctx.request_repaint();
                }
            }

            // Add progress bar at the bottom with proper spacing
            ui.add_space(ui.available_height() - 10.0);
            let progress = (self.current_index as f32 + 1.0) / self.images.len() as f32;
            ui.add(egui::ProgressBar::new(progress)
                .show_percentage()
                .desired_width(ui.available_width())
                .desired_height(10.0));
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Err(e) = self.save_state() {
            eprintln!("Failed to save application state: {}", e);
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <directory>", args[0]);
        std::process::exit(1);
    }

    let directory = Path::new(&args[1]);
    if !directory.exists() || !directory.is_dir() {
        anyhow::bail!("Invalid directory path");
    }

    let viewer = ImageViewer::new(directory)?;

    let options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([600.0, 400.0]) // Minimum size for usability
            .with_resizable(true),
        vsync: true, // Enable vsync for smoother rendering
        multisampling: 4, // Enable 4x MSAA for better quality
        depth_buffer: 0, // We don't need depth buffer for 2D images
        stencil_buffer: 0, // We don't need stencil buffer
        hardware_acceleration: eframe::HardwareAcceleration::Required, // Force GPU acceleration
        ..Default::default()
    };

    eframe::run_native(
        "Image Viewer - GPU Accelerated",
        options,
        Box::new(|cc| {
            // Configure egui for better performance
            cc.egui_ctx.set_visuals(egui::Visuals {
                window_corner_radius: CornerRadius::same(5),
                ..egui::Visuals::dark()
            });

            // Set texture allocation options for better GPU performance
            cc.egui_ctx.set_style(egui::Style {
                animation_time: 0.1, // Faster animations
                ..egui::Style::default()
            });

            Ok(Box::new(viewer))
        }),
    ).map_err(|e| anyhow::anyhow!("Failed to run application: {}", e))?;
    Ok(())
}