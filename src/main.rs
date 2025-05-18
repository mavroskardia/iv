use anyhow::Result;
use eframe::egui::{self, ViewportBuilder, ViewportCommand};
use image::DynamicImage;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use walkdir::WalkDir;
use std::fs;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Default)]
struct AppState {
    last_positions: HashMap<String, usize>,
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

        // Create favorites and deleted directories
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path.parent()
            .ok_or_else(|| anyhow::anyhow!("Could not get executable directory"))?;
        let favorites_dir = exe_dir.join("favorites");
        let deleted_dir = exe_dir.join("deleted");

        fs::create_dir_all(&favorites_dir)?;
        fs::create_dir_all(&deleted_dir)?;

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
        };

        // Load initial images
        viewer.load_initial_images()?;
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
        Ok(())
    }
}

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_loaded_images();

        // Update flash animation
        if self.flash_state > 0.0 {
            self.flash_state = (self.flash_state - 0.05).max(0.0);
            ctx.request_repaint();
        }

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
                let scale = (available_size.x / size[0]).min(available_size.y / size[1]);
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
        Box::new(|_cc| Box::new(viewer)),
    ).map_err(|e| anyhow::anyhow!("Failed to run application: {}", e))?;
    Ok(())
}