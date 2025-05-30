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
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use sha2::{Sha256, Digest};
use hex;
use rand::prelude::*;

#[cfg(feature = "gpu")]
use candle_core::{Device as CandleDevice, Tensor as CandleTensor, DType};
#[cfg(feature = "gpu")]
use candle_nn::{Linear, Module, VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};

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

#[derive(Serialize, Deserialize, Clone)]
struct SimpleNeuralNetwork {
    weights1: Array2<f32>, // 512 -> 128
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
        // Xavier initialization using ndarray-rand
        let weights1 = Array2::random((512, 128), Uniform::new(-0.1, 0.1));
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
        if features.len() != 512 {
            return Err(anyhow::anyhow!("Expected 512 features, got {}", features.len()));
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
                if sample.features.len() == 512 {
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
            shuffled_data.shuffle(&mut thread_rng());

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
            shuffled_data.shuffle(&mut thread_rng());

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
            if sample.features.len() != 512 {
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

        let layer1 = candle_nn::linear(512, 128, vs.pp("layer1"))?;
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
        if features.len() != 512 {
            return Err(anyhow::anyhow!("Expected 512 features, got {}", features.len()));
        }

        let input = CandleTensor::from_slice(features, (1, 512), &self.device)?;
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
                if sample.features.len() == 512 {
                    if let Ok(prediction) = self.predict(&sample.features) {
                        if prediction == sample.rating {
                            correct_predictions += 1;
                        }
                    }

                    // Calculate loss
                    let input = CandleTensor::from_slice(&sample.features, (1, 512), &self.device)?;
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
            shuffled_data.shuffle(&mut thread_rng());

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
            if sample.features.len() == 512 {
                inputs.extend_from_slice(&sample.features);
                targets.push((sample.rating - 1) as usize);
            }
        }

        if inputs.is_empty() {
            return Ok(());
        }

        let input_tensor = CandleTensor::from_slice(&inputs, (batch_size, 512), &self.device)?;
        let output = self.forward(&input_tensor)?;

        // Create target tensor
        let mut target_data = vec![0.0f32; batch_size * 5];
        for (i, &target_class) in targets.iter().enumerate() {
            target_data[i * 5 + target_class] = 1.0;
        }
        let target_tensor = CandleTensor::from_slice(&target_data, (batch_size, 5), &self.device)?;

        // Cross-entropy loss
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
                println!("Windows GNU detected: CUDA not supported, using CPU-only mode");
                println!("For GPU acceleration on Windows, please use the MSVC toolchain:");
                println!("  rustup default stable-x86_64-pc-windows-msvc");
                println!("  cargo build --features cuda --target x86_64-pc-windows-msvc");
                (None, false)
            }
            #[cfg(not(all(target_os = "windows", target_env = "gnu")))]
            {
                match CandleDevice::new_cuda(0) {
                    Ok(device) => {
                        match GpuNeuralNetwork::new(device) {
                            Ok(gpu_net) => {
                                println!("GPU acceleration enabled with CUDA");
                                (Some(gpu_net), true)
                            }
                            Err(e) => {
                                eprintln!("Failed to initialize GPU network: {}, falling back to CPU", e);
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
                                        println!("GPU acceleration enabled with Metal");
                                        (Some(gpu_net), true)
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to initialize GPU network: {}, falling back to CPU", e);
                                        (None, false)
                                    }
                                }
                            }
                            Err(_) => {
                                println!("No GPU available, using CPU");
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
    loaded_images: Vec<Option<DynamicImage>>,
    image_textures: Vec<Option<egui::TextureHandle>>,
    image_loading_tx: Sender<(usize, PathBuf)>,
    image_loading_rx: Receiver<(usize, DynamicImage)>,
    favorites_dir: PathBuf,
    deleted_dir: PathBuf,
    favorite_counter: usize,
    flash_state: f32, // 0.0 to 1.0, where 1.0 is full flash
    flash_color: egui::Color32, // Color of the flash effect
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
    // Background processing
    feature_extraction_tx: Sender<(String, DynamicImage)>,
    feature_extraction_rx: Receiver<(String, Vec<f32>)>,
    prediction_tx: Sender<(String, Vec<f32>)>,
    prediction_rx: Receiver<(String, u8)>,
    pending_suggestions: HashMap<String, bool>, // Track which images are being processed
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

        let (tx, rx) = channel();
        let (loading_tx, loading_rx) = channel();

        // Start image loading thread
        thread::spawn(move || {
            while let Ok((index, path)) = rx.recv() {
                if let Ok(img) = image::open(&path) {
                    let _ = loading_tx.send((index, img));
                }
            }
        });

        // Background processing channels
        let (feature_tx, feature_rx) = channel::<(String, DynamicImage)>();
        let (feature_result_tx, feature_result_rx) = channel::<(String, Vec<f32>)>();
        let (prediction_tx, prediction_rx) = channel::<(String, Vec<f32>)>();
        let (prediction_result_tx, prediction_result_rx) = channel::<(String, u8)>();

        // Shared predictor for background thread
        let shared_predictor: Arc<Mutex<Option<RatingPredictor>>> = Arc::new(Mutex::new(None));
        let thread_predictor = shared_predictor.clone();

        // Start feature extraction thread
        thread::spawn(move || {
            while let Ok((hash, image)) = feature_rx.recv() {
                let features = Self::extract_features_static(&image);
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
            loaded_images: vec![None; 5], // Current + 2 before + 2 after
            image_textures: vec![None; 5],
            image_loading_tx: tx,
            image_loading_rx: loading_rx,
            favorites_dir,
            deleted_dir,
            favorite_counter,
            flash_state: 0.0,
            flash_color: egui::Color32::WHITE, // Default flash color
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
            // Background processing
            feature_extraction_tx: feature_tx,
            feature_extraction_rx: feature_result_rx,
            prediction_tx: prediction_tx,
            prediction_rx: prediction_result_rx,
            pending_suggestions: HashMap::new(),
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
        // Load current image and 2 before and after
        for i in 0..5 {
            let index = if i < 2 {
                self.current_index.saturating_sub(2 - i)
            } else {
                self.current_index + (i - 2)
            };

            if index < self.images.len() {
                self.load_image(index)?;
            }
        }
        Ok(())
    }

    fn load_image(&mut self, index: usize) -> Result<()> {
        if index < self.images.len() {
            let path = self.images[index].clone();
            self.image_loading_tx.send((index, path))?;
        }
        Ok(())
    }

    fn update_loaded_images(&mut self) {
        while let Ok((index, img)) = self.image_loading_rx.try_recv() {
            let cache_index = if index < self.current_index {
                if self.current_index - index <= 2 {
                    2 - (self.current_index - index)
                } else {
                    continue;
                }
            } else if index > self.current_index {
                if index - self.current_index <= 2 {
                    2 + (index - self.current_index)
                } else {
                    continue;
                }
            } else {
                2
            };

            if cache_index < self.loaded_images.len() {
                self.loaded_images[cache_index] = Some(img);
                // Clear the corresponding texture to force a reload
                self.image_textures[cache_index] = None;
            }
        }
    }

    fn next_image(&mut self) -> Result<()> {
        if self.current_index + 1 < self.images.len() {
            self.current_index += 1;
            // Clear the current textures to force reload
            self.image_textures = vec![None; 5];
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
            // Clear the current textures to force reload
            self.image_textures = vec![None; 5];
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

        self.flash_state = 1.0;
        self.flash_color = egui::Color32::WHITE;

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
        self.loaded_images = vec![None; 5];
        self.image_textures = vec![None; 5];

        // If we've deleted the last image, go to the previous one
        if self.current_index >= self.images.len() && !self.images.is_empty() {
            self.current_index = self.images.len() - 1;
        }

        // Set flash state to red
        self.flash_state = 1.0;
        self.flash_color = egui::Color32::RED;

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

    fn extract_features(&self, image: &DynamicImage) -> Vec<f32> {
        Self::extract_features_static(image)
    }

    fn extract_features_static(image: &DynamicImage) -> Vec<f32> {
        let mut features = Vec::with_capacity(512);

        // Convert to different formats for analysis
        let rgb = image.to_rgb8();
        let gray = image.to_luma8();
        let (width, height) = (rgb.width(), rgb.height());

        // 1. Color histogram features (96 features: 32 each for R, G, B)
        let mut r_hist = vec![0u32; 32];
        let mut g_hist = vec![0u32; 32];
        let mut b_hist = vec![0u32; 32];

        for pixel in rgb.pixels() {
            let r_bin = (pixel[0] as usize * 31) / 255;
            let g_bin = (pixel[1] as usize * 31) / 255;
            let b_bin = (pixel[2] as usize * 31) / 255;
            r_hist[r_bin] += 1;
            g_hist[g_bin] += 1;
            b_hist[b_bin] += 1;
        }

        let total_pixels = (width * height) as f32;
        for &count in &r_hist {
            features.push(count as f32 / total_pixels);
        }
        for &count in &g_hist {
            features.push(count as f32 / total_pixels);
        }
        for &count in &b_hist {
            features.push(count as f32 / total_pixels);
        }

        // 2. Brightness and contrast statistics (8 features)
        let gray_pixels: Vec<f32> = gray.pixels().map(|p| p[0] as f32 / 255.0).collect();
        let mean_brightness = gray_pixels.iter().sum::<f32>() / gray_pixels.len() as f32;
        let brightness_variance = gray_pixels.iter()
            .map(|&x| (x - mean_brightness).powi(2))
            .sum::<f32>() / gray_pixels.len() as f32;
        let brightness_std = brightness_variance.sqrt();

        // Contrast (RMS contrast)
        let rms_contrast = brightness_std;

        // Histogram-based contrast (difference between 95th and 5th percentiles)
        let mut sorted_gray = gray_pixels.clone();
        sorted_gray.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p5 = sorted_gray[(sorted_gray.len() as f32 * 0.05) as usize];
        let p95 = sorted_gray[(sorted_gray.len() as f32 * 0.95) as usize];
        let percentile_contrast = p95 - p5;

        features.extend_from_slice(&[
            mean_brightness,
            brightness_variance,
            brightness_std,
            rms_contrast,
            percentile_contrast,
            sorted_gray[sorted_gray.len() / 4], // 25th percentile
            sorted_gray[sorted_gray.len() / 2], // median
            sorted_gray[sorted_gray.len() * 3 / 4], // 75th percentile
        ]);

        // 3. Edge density and sharpness (Sobel operator) (4 features)
        let mut edge_magnitude_sum = 0.0;
        let mut strong_edges = 0;
        let mut edge_magnitudes = Vec::new();

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let get_gray = |x: u32, y: u32| -> f32 {
                    gray.get_pixel(x, y)[0] as f32 / 255.0
                };

                // Sobel operators
                let gx = -get_gray(x-1, y-1) - 2.0*get_gray(x-1, y) - get_gray(x-1, y+1)
                        + get_gray(x+1, y-1) + 2.0*get_gray(x+1, y) + get_gray(x+1, y+1);
                let gy = -get_gray(x-1, y-1) - 2.0*get_gray(x, y-1) - get_gray(x+1, y-1)
                        + get_gray(x-1, y+1) + 2.0*get_gray(x, y+1) + get_gray(x+1, y+1);

                let magnitude = (gx*gx + gy*gy).sqrt();
                edge_magnitude_sum += magnitude;
                edge_magnitudes.push(magnitude);

                if magnitude > 0.1 { // Threshold for strong edges
                    strong_edges += 1;
                }
            }
        }

        let edge_density = edge_magnitude_sum / ((width - 2) * (height - 2)) as f32;
        let strong_edge_ratio = strong_edges as f32 / ((width - 2) * (height - 2)) as f32;

        // Edge magnitude statistics
        edge_magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let edge_median = edge_magnitudes[edge_magnitudes.len() / 2];
        let edge_95th = edge_magnitudes[(edge_magnitudes.len() as f32 * 0.95) as usize];

        features.extend_from_slice(&[edge_density, strong_edge_ratio, edge_median, edge_95th]);

        // 4. Color saturation and vibrancy (6 features)
        let mut saturation_sum = 0.0;
        let mut hue_distribution = vec![0u32; 12]; // 12 hue bins
        let mut saturated_pixels = 0;

        for pixel in rgb.pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            let max_val = r.max(g).max(b);
            let min_val = r.min(g).min(b);
            let delta = max_val - min_val;

            // Saturation
            let saturation = if max_val > 0.0 { delta / max_val } else { 0.0 };
            saturation_sum += saturation;

            if saturation > 0.3 {
                saturated_pixels += 1;

                // Hue calculation for saturated pixels
                if delta > 0.0 {
                    let hue = if max_val == r {
                        ((g - b) / delta) % 6.0
                    } else if max_val == g {
                        (b - r) / delta + 2.0
                    } else {
                        (r - g) / delta + 4.0
                    };
                    let hue_normalized = (hue * 60.0 + 360.0) % 360.0;
                    let hue_bin = (hue_normalized / 30.0) as usize % 12;
                    hue_distribution[hue_bin] += 1;
                }
            }
        }

        let avg_saturation = saturation_sum / total_pixels;
        let saturated_pixel_ratio = saturated_pixels as f32 / total_pixels;

        // Hue diversity (entropy-like measure)
        let mut hue_entropy = 0.0;
        if saturated_pixels > 0 {
            for &count in &hue_distribution {
                if count > 0 {
                    let p = count as f32 / saturated_pixels as f32;
                    hue_entropy -= p * p.log2();
                }
            }
        }

        features.extend_from_slice(&[
            avg_saturation,
            saturated_pixel_ratio,
            hue_entropy,
            hue_distribution.iter().max().unwrap_or(&0).clone() as f32 / saturated_pixels.max(1) as f32, // dominant hue strength
        ]);

        // 5. Texture analysis using Local Binary Patterns (simplified) (16 features)
        let mut lbp_histogram = vec![0u32; 16]; // Simplified 4-bit LBP

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let center = gray.get_pixel(x, y)[0];
                let mut lbp_code = 0u8;

                // 4-neighbor LBP (simplified)
                let neighbors = [
                    gray.get_pixel(x, y-1)[0], // top
                    gray.get_pixel(x+1, y)[0], // right
                    gray.get_pixel(x, y+1)[0], // bottom
                    gray.get_pixel(x-1, y)[0], // left
                ];

                for (i, &neighbor) in neighbors.iter().enumerate() {
                    if neighbor >= center {
                        lbp_code |= 1 << i;
                    }
                }

                lbp_histogram[lbp_code as usize] += 1;
            }
        }

        let lbp_total = ((width - 2) * (height - 2)) as f32;
        for &count in &lbp_histogram {
            features.push(count as f32 / lbp_total);
        }

        // 6. Noise estimation (4 features)
        // High-frequency noise estimation using Laplacian
        let mut noise_sum = 0.0;
        let mut high_freq_energy = 0.0;

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let center = gray.get_pixel(x, y)[0] as f32;
                let laplacian = -4.0 * center
                    + gray.get_pixel(x-1, y)[0] as f32
                    + gray.get_pixel(x+1, y)[0] as f32
                    + gray.get_pixel(x, y-1)[0] as f32
                    + gray.get_pixel(x, y+1)[0] as f32;

                noise_sum += laplacian.abs();
                high_freq_energy += laplacian * laplacian;
            }
        }

        let noise_level = noise_sum / ((width - 2) * (height - 2)) as f32;
        let high_freq_avg = high_freq_energy / ((width - 2) * (height - 2)) as f32;

        features.extend_from_slice(&[
            noise_level,
            high_freq_avg,
            noise_level / (mean_brightness + 0.001), // noise-to-signal ratio
            high_freq_avg.sqrt(), // RMS high frequency
        ]);

        // 7. Composition features (8 features)
        // Rule of thirds analysis
        let third_w = width / 3;
        let third_h = height / 3;

        let mut region_brightness = vec![0.0f32; 9];
        let mut region_counts = vec![0u32; 9];

        for y in 0..height {
            for x in 0..width {
                let region_x = (x / third_w).min(2) as usize;
                let region_y = (y / third_h).min(2) as usize;
                let region_idx = region_y * 3 + region_x;

                region_brightness[region_idx] += gray.get_pixel(x, y)[0] as f32;
                region_counts[region_idx] += 1;
            }
        }

        for i in 0..9 {
            if region_counts[i] > 0 {
                region_brightness[i] /= region_counts[i] as f32;
            }
        }

        // Center vs edges brightness difference
        let center_brightness = region_brightness[4]; // center region
        let edge_brightness = (region_brightness[0] + region_brightness[2] +
                              region_brightness[6] + region_brightness[8]) / 4.0; // corners
        let center_edge_diff = (center_brightness - edge_brightness).abs();

        features.extend_from_slice(&region_brightness);
        features.push(center_edge_diff);

        // 8. Aspect ratio and resolution features (4 features)
        let aspect_ratio = width as f32 / height as f32;
        let resolution_score = (width * height) as f32 / (1920.0 * 1080.0); // normalized to 1080p
        let width_score = width as f32 / 1920.0;
        let height_score = height as f32 / 1080.0;

        features.extend_from_slice(&[aspect_ratio, resolution_score, width_score, height_score]);

        // 9. Shape and geometry analysis (32 features)
        let shape_features = Self::extract_shape_features(&gray, width, height);
        features.extend_from_slice(&shape_features);

        // Ensure we have exactly 512 features
        features.resize(512, 0.0);

        // Normalize features to prevent any single feature from dominating
        let max_val = features.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        if max_val > 0.0 {
            for feature in &mut features {
                *feature /= max_val;
            }
        }

        features
    }

    fn extract_shape_features(gray: &image::GrayImage, width: u32, height: u32) -> Vec<f32> {
        let mut shape_features = Vec::with_capacity(32);

        // 1. Corner detection using Harris corner detector (simplified) (4 features)
        let mut corner_count = 0;
        let mut corner_strength_sum = 0.0;
        let mut corner_responses = Vec::new();

        for y in 2..(height - 2) {
            for x in 2..(width - 2) {
                // Simplified Harris corner response
                let mut ix = 0.0f32;
                let mut iy = 0.0f32;
                let mut ixy = 0.0f32;

                // Calculate gradients in 3x3 window
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let px = gray.get_pixel(x + dx as u32, y + dy as u32)[0] as f32;
                        let gx = if dx != 0 { dx as f32 * px } else { 0.0 };
                        let gy = if dy != 0 { dy as f32 * px } else { 0.0 };

                        ix += gx * gx;
                        iy += gy * gy;
                        ixy += gx * gy;
                    }
                }

                // Harris response
                let det = ix * iy - ixy * ixy;
                let trace = ix + iy;
                let response = det - 0.04 * trace * trace;

                if response > 0.01 {
                    corner_count += 1;
                    corner_strength_sum += response;
                    corner_responses.push(response);
                }
            }
        }

        let corner_density = corner_count as f32 / ((width - 4) * (height - 4)) as f32;
        let avg_corner_strength = if corner_count > 0 { corner_strength_sum / corner_count as f32 } else { 0.0 };

        corner_responses.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let max_corner_strength = corner_responses.first().copied().unwrap_or(0.0);
        let corner_strength_variance = if corner_responses.len() > 1 {
            let mean = avg_corner_strength;
            corner_responses.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / corner_responses.len() as f32
        } else { 0.0 };

        shape_features.extend_from_slice(&[corner_density, avg_corner_strength, max_corner_strength, corner_strength_variance]);

        // 2. Line detection using Hough transform (simplified) (8 features)
        let mut horizontal_lines = 0;
        let mut vertical_lines = 0;
        let mut diagonal_lines = 0;
        let mut line_strength_sum = 0.0;

        // Simplified line detection by analyzing edge directions
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let gx = gray.get_pixel(x + 1, y)[0] as f32 - gray.get_pixel(x - 1, y)[0] as f32;
                let gy = gray.get_pixel(x, y + 1)[0] as f32 - gray.get_pixel(x, y - 1)[0] as f32;

                let magnitude = (gx * gx + gy * gy).sqrt();
                if magnitude > 20.0 {
                    let angle = gy.atan2(gx).abs();
                    line_strength_sum += magnitude;

                    if angle < 0.3 || angle > 2.8 { // ~horizontal
                        horizontal_lines += 1;
                    } else if angle > 1.3 && angle < 1.8 { // ~vertical
                        vertical_lines += 1;
                    } else { // diagonal
                        diagonal_lines += 1;
                    }
                }
            }
        }

        let total_pixels = ((width - 2) * (height - 2)) as f32;
        let horizontal_line_density = horizontal_lines as f32 / total_pixels;
        let vertical_line_density = vertical_lines as f32 / total_pixels;
        let diagonal_line_density = diagonal_lines as f32 / total_pixels;
        let avg_line_strength = line_strength_sum / total_pixels;
        let line_orientation_ratio = if vertical_lines > 0 { horizontal_lines as f32 / vertical_lines as f32 } else { 0.0 };
        let total_line_density = (horizontal_lines + vertical_lines + diagonal_lines) as f32 / total_pixels;
        let line_complexity = if total_line_density > 0.0 { diagonal_line_density / total_line_density } else { 0.0 };
        let line_regularity = if total_line_density > 0.0 {
            (horizontal_line_density + vertical_line_density) / total_line_density
        } else { 0.0 };

        shape_features.extend_from_slice(&[
            horizontal_line_density, vertical_line_density, diagonal_line_density, avg_line_strength,
            line_orientation_ratio, total_line_density, line_complexity, line_regularity
        ]);

        // 3. Symmetry analysis (8 features)
        let center_x = width / 2;
        let center_y = height / 2;

        // Horizontal symmetry
        let mut h_symmetry_error = 0.0;
        let mut h_symmetry_count = 0;
        for y in 0..height {
            for x in 0..center_x {
                let mirror_x = width - 1 - x;
                if mirror_x < width {
                    let left_pixel = gray.get_pixel(x, y)[0] as f32;
                    let right_pixel = gray.get_pixel(mirror_x, y)[0] as f32;
                    h_symmetry_error += (left_pixel - right_pixel).abs();
                    h_symmetry_count += 1;
                }
            }
        }
        let horizontal_symmetry = if h_symmetry_count > 0 {
            1.0 - (h_symmetry_error / (h_symmetry_count as f32 * 255.0))
        } else { 0.0 };

        // Vertical symmetry
        let mut v_symmetry_error = 0.0;
        let mut v_symmetry_count = 0;
        for y in 0..center_y {
            for x in 0..width {
                let mirror_y = height - 1 - y;
                if mirror_y < height {
                    let top_pixel = gray.get_pixel(x, y)[0] as f32;
                    let bottom_pixel = gray.get_pixel(x, mirror_y)[0] as f32;
                    v_symmetry_error += (top_pixel - bottom_pixel).abs();
                    v_symmetry_count += 1;
                }
            }
        }
        let vertical_symmetry = if v_symmetry_count > 0 {
            1.0 - (v_symmetry_error / (v_symmetry_count as f32 * 255.0))
        } else { 0.0 };

        // Diagonal symmetries (simplified)
        let mut d1_symmetry_error = 0.0;
        let mut d2_symmetry_error = 0.0;
        let mut d_symmetry_count = 0;

        let min_dim = width.min(height);
        for i in 0..min_dim {
            for j in 0..min_dim {
                if i < width && j < height && j < width && i < height {
                    let p1 = gray.get_pixel(i, j)[0] as f32;
                    let p2 = gray.get_pixel(j, i)[0] as f32; // Main diagonal
                    let p3 = gray.get_pixel(min_dim - 1 - i, j)[0] as f32;
                    let p4 = gray.get_pixel(min_dim - 1 - j, i)[0] as f32; // Anti-diagonal

                    d1_symmetry_error += (p1 - p2).abs();
                    d2_symmetry_error += (p3 - p4).abs();
                    d_symmetry_count += 1;
                }
            }
        }

        let diagonal1_symmetry = if d_symmetry_count > 0 {
            1.0 - (d1_symmetry_error / (d_symmetry_count as f32 * 255.0))
        } else { 0.0 };
        let diagonal2_symmetry = if d_symmetry_count > 0 {
            1.0 - (d2_symmetry_error / (d_symmetry_count as f32 * 255.0))
        } else { 0.0 };

        let overall_symmetry = (horizontal_symmetry + vertical_symmetry + diagonal1_symmetry + diagonal2_symmetry) / 4.0;
        let symmetry_variance = [horizontal_symmetry, vertical_symmetry, diagonal1_symmetry, diagonal2_symmetry]
            .iter().map(|&x| (x - overall_symmetry).powi(2)).sum::<f32>() / 4.0;
        let max_symmetry = [horizontal_symmetry, vertical_symmetry, diagonal1_symmetry, diagonal2_symmetry]
            .iter().fold(0.0f32, |acc, &x| acc.max(x));
        let symmetry_bias = if overall_symmetry > 0.0 { max_symmetry / overall_symmetry } else { 0.0 };

        shape_features.extend_from_slice(&[
            horizontal_symmetry, vertical_symmetry, diagonal1_symmetry, diagonal2_symmetry,
            overall_symmetry, symmetry_variance, max_symmetry, symmetry_bias
        ]);

        // 4. Geometric shape detection (12 features)
        // Circle detection using Hough circle transform (simplified)
        let mut circle_score = 0.0;

        // Analyze edge curvature for circles
        let mut curvature_sum = 0.0;
        let mut curvature_count = 0;

        for y in 2..(height - 2) {
            for x in 2..(width - 2) {
                // Calculate local curvature using second derivatives
                let center = gray.get_pixel(x, y)[0] as f32;
                let left = gray.get_pixel(x - 1, y)[0] as f32;
                let right = gray.get_pixel(x + 1, y)[0] as f32;
                let top = gray.get_pixel(x, y - 1)[0] as f32;
                let bottom = gray.get_pixel(x, y + 1)[0] as f32;

                let second_deriv_x = left - 2.0 * center + right;
                let second_deriv_y = top - 2.0 * center + bottom;
                let curvature = (second_deriv_x.abs() + second_deriv_y.abs()) / 2.0;

                if curvature > 5.0 {
                    curvature_sum += curvature;
                    curvature_count += 1;

                    // High curvature suggests circular features
                    if curvature > 15.0 {
                        circle_score += 1.0;
                    }
                }
            }
        }

        circle_score /= total_pixels;
        let avg_curvature = if curvature_count > 0 { curvature_sum / curvature_count as f32 } else { 0.0 };

        // Rectangle detection (look for right angles and parallel lines)
        let rectangle_score = (horizontal_line_density + vertical_line_density) * line_regularity;

        // Triangle detection (look for three dominant edge directions)
        let edge_direction_entropy = if total_line_density > 0.0 {
            let h_ratio = horizontal_line_density / total_line_density;
            let v_ratio = vertical_line_density / total_line_density;
            let d_ratio = diagonal_line_density / total_line_density;

            let mut entropy = 0.0;
            if h_ratio > 0.0 { entropy -= h_ratio * h_ratio.log2(); }
            if v_ratio > 0.0 { entropy -= v_ratio * v_ratio.log2(); }
            if d_ratio > 0.0 { entropy -= d_ratio * d_ratio.log2(); }
            entropy
        } else { 0.0 };

        let triangle_score = edge_direction_entropy * diagonal_line_density;

        // Additional geometric features
        let shape_complexity = corner_density * total_line_density;
        let geometric_regularity = (rectangle_score + overall_symmetry) / 2.0;
        let organic_vs_geometric = circle_score / (rectangle_score + 0.001);
        let edge_coherence = avg_line_strength / (avg_curvature + 1.0);
        let structural_balance = overall_symmetry * geometric_regularity;
        let shape_diversity = edge_direction_entropy;
        let angular_features = corner_density / (circle_score + 0.001);
        let geometric_harmony = (overall_symmetry + geometric_regularity + line_regularity) / 3.0;

        shape_features.extend_from_slice(&[
            circle_score, rectangle_score, triangle_score, avg_curvature,
            shape_complexity, geometric_regularity, organic_vs_geometric, edge_coherence,
            structural_balance, shape_diversity, angular_features, geometric_harmony
        ]);

        // Ensure we have exactly 32 shape features
        shape_features.resize(32, 0.0);
        shape_features
    }

    fn rate_current_image(&mut self, rating: u8) -> Result<()> {
        if let Some(image) = &self.loaded_images[2] {
            let hash = self.compute_image_hash(image);
            let features = if let Some(cached) = self.image_features_cache.get(&hash) {
                cached.clone()
            } else {
                let features = self.extract_features(image);
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

            // Flash effect for rating
            self.flash_state = 1.0;
            self.flash_color = match rating {
                1 => egui::Color32::from_rgb(255, 100, 100), // Red
                2 => egui::Color32::from_rgb(255, 165, 0),   // Orange
                3 => egui::Color32::from_rgb(255, 255, 100), // Yellow
                4 => egui::Color32::from_rgb(144, 238, 144), // Light green
                5 => egui::Color32::from_rgb(100, 255, 100), // Green
                _ => egui::Color32::WHITE,
            };
        }

        Ok(())
    }

    fn update_suggestion(&mut self) {
        // Process any completed background work first
        self.process_background_results();

        if let Some(image) = &self.loaded_images[2] {
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
                // No cached features, start background extraction if not already pending
                if !self.pending_suggestions.contains_key(&hash) {
                    self.pending_suggestions.insert(hash.clone(), true);
                    let hash_clone = hash.clone();
                    if let Err(e) = self.feature_extraction_tx.send((hash_clone, image.clone())) {
                        eprintln!("Failed to send image for feature extraction: {}", e);
                        self.pending_suggestions.remove(&hash);
                    }
                }

                // Clear current state while processing
                self.suggested_rating = None;
                self.current_rating = None;
            }
        }
    }

    fn process_background_results(&mut self) {
        // Process completed feature extractions
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
            if let Some(current_image) = &self.loaded_images[2] {
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

        // Preemptively extract features for nearby images
        self.preextract_nearby_features();
    }

    fn preextract_nearby_features(&mut self) {
        // Extract features for next 2 images if they're loaded and not already cached/pending
        for i in 3..5 { // indices 3 and 4 are the next 2 images
            if let Some(image) = &self.loaded_images[i] {
                let hash = self.compute_image_hash(image);

                // Only start extraction if not already cached or pending
                if !self.image_features_cache.contains_key(&hash) &&
                   !self.pending_suggestions.contains_key(&hash) {
                    self.pending_suggestions.insert(hash.clone(), true);
                    if let Err(_) = self.feature_extraction_tx.send((hash.clone(), image.clone())) {
                        self.pending_suggestions.remove(&hash);
                    }
                }
            }
        }
    }
}

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update window title with current filename
        if let Some(current_path) = self.images.get(self.current_index) {
            if let Some(filename) = current_path.file_name() {
                if let Some(filename_str) = filename.to_str() {
                    ctx.send_viewport_cmd(ViewportCommand::Title(format!("{} | IV", filename_str)));
                }
            }
        }

        self.update_loaded_images();

        // Update flash animation
        if self.flash_state > 0.0 {
            self.flash_state = (self.flash_state - 0.05).max(0.0);
            ctx.request_repaint();
        }

        // Update rating suggestions when images are loaded
        self.update_suggestion();

        // Process background results to keep UI responsive
        self.process_background_results();

        // Request continuous updates to catch keyboard events
        ctx.request_repaint();

        // Handle keyboard input
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            let _ = self.next_image();
            ctx.request_repaint();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            let _ = self.previous_image();
            ctx.request_repaint();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            ctx.send_viewport_cmd(ViewportCommand::Close);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Plus)) {
            if let Err(e) = self.copy_to_favorites() {
                eprintln!("Failed to copy image to favorites: {}", e);
            }
            ctx.request_repaint();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            if let Err(e) = self.move_to_deleted() {
                eprintln!("Failed to move image to deleted folder: {}", e);
            }
            ctx.request_repaint();
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
                    // Automatically advance to next image after rating
                    let _ = self.next_image();
                }
                ctx.request_repaint();
            }
        }

        // Debug feature extraction (press D key)
        if ctx.input(|i| i.key_pressed(egui::Key::D)) {
            if let Some(image) = &self.loaded_images[2] {
                let features = self.extract_features(image);
                println!("\n=== DEBUG: Feature Analysis ===");
                println!("Image size: {}x{}", image.width(), image.height());

                // Color and brightness features
                println!("Brightness (mean): {:.3}", features[96]); // First brightness feature
                println!("Contrast (RMS): {:.3}", features[99]); // RMS contrast
                println!("Average saturation: {:.3}", features[120]); // Saturation

                // Edge and texture features
                println!("Edge density: {:.3}", features[104]); // Edge density
                println!("Strong edge ratio: {:.3}", features[105]); // Strong edges
                println!("Noise level: {:.3}", features[136]); // Noise

                // Composition features
                println!("Center-edge brightness diff: {:.3}", features[152]); // Composition
                println!("Aspect ratio: {:.3}", features[153]); // Aspect ratio
                println!("Resolution score: {:.3}", features[154]); // Resolution

                // New shape features (starting at index 156)
                println!("\n--- Shape Analysis ---");
                println!("Corner density: {:.3}", features[156]); // Corner density
                println!("Horizontal lines: {:.3}", features[160]); // Horizontal line density
                println!("Vertical lines: {:.3}", features[161]); // Vertical line density
                println!("Line regularity: {:.3}", features[167]); // Line regularity
                println!("Horizontal symmetry: {:.3}", features[168]); // Horizontal symmetry
                println!("Vertical symmetry: {:.3}", features[169]); // Vertical symmetry
                println!("Overall symmetry: {:.3}", features[172]); // Overall symmetry
                println!("Circle score: {:.3}", features[176]); // Circle detection
                println!("Rectangle score: {:.3}", features[177]); // Rectangle detection
                println!("Geometric regularity: {:.3}", features[181]); // Geometric regularity
                println!("Organic vs geometric: {:.3}", features[182]); // Organic vs geometric
                println!("Geometric harmony: {:.3}", features[187]); // Geometric harmony

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

        // Handle mouse input
        if ctx.input(|i| i.pointer.primary_clicked()) {
            let _ = self.next_image();
            ctx.request_repaint();
        }
        if ctx.input(|i| i.pointer.secondary_clicked()) {
            let _ = self.previous_image();
            ctx.request_repaint();
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

                // Show GPU acceleration status
                if let Some(ref predictor) = self.predictor {
                    let performance_info = predictor.get_performance_info();
                    let color = if performance_info.contains("GPU") {
                        egui::Color32::from_rgb(100, 255, 100) // Green for GPU
                    } else {
                        egui::Color32::from_rgb(255, 255, 100) // Yellow for CPU
                    };
                    ui.colored_label(color, performance_info);
                }
            });

            ui.separator();

            if let Some(img) = &self.loaded_images[2] {
                let size = [img.width() as f32, img.height() as f32];
                let texture = self.image_textures[2].get_or_insert_with(|| {
                    let color_image = egui::ColorImage::from_rgba_unmultiplied(
                        [img.width() as usize, img.height() as usize],
                        &img.to_rgba8().into_raw(),
                    );
                    ui.ctx().load_texture(
                        "current_image",
                        color_image,
                        egui::TextureOptions::default(),
                    )
                });

                let available_size = ui.available_size();
                let scale = (available_size.x / size[0]).min((available_size.y - 40.0) / size[1]); // Reserve space for progress bar
                let scaled_size = egui::vec2(size[0] * scale, size[1] * scale);

                let response = ui.centered_and_justified(|ui| {
                    let response = ui.add(egui::Image::new((texture.id(), scaled_size))
                        .sense(egui::Sense::click_and_drag()));

                    // Draw flash overlay if active
                    if self.flash_state > 0.0 {
                        let rect = response.rect;
                        let alpha = (self.flash_state * 0.5 * 255.0) as u8;
                        let flash_color = match self.flash_color {
                            egui::Color32::WHITE => egui::Color32::from_white_alpha(alpha),
                            egui::Color32::RED => egui::Color32::from_rgba_premultiplied(255, 0, 0, alpha),
                            _ => egui::Color32::from_white_alpha(alpha),
                        };
                        ui.painter().rect_filled(
                            rect,
                            0.0,
                            flash_color,
                        );
                    }

                    response
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
            .with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(viewer))),
    ).map_err(|e| anyhow::anyhow!("Failed to run application: {}", e))?;
    Ok(())
}