use eframe::egui;
use egui::ColorImage;
use egui::Response;
use egui::TextureHandle;
use glam::{vec2, vec3};
use rand::{thread_rng, Rng};
use std::time::Instant;

use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 1000.0]),
        ..Default::default()
    };

    let thread_pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_| {
            Box::new(MyApp {
                accumulate: false,
                pool: thread_pool,
                moved: false,
                image: egui::ColorImage::from_rgb([0, 0], &[]),
            })
        }),
    )
}

struct MyApp {
    accumulate: bool,
    pool: ThreadPool,
    moved: bool,
    image: ColorImage,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let start_time = Instant::now();

        // Set the entire background color
        let visual = egui::Visuals::dark();
        ctx.set_visuals(visual);

        let size: egui::Vec2 = ctx.available_rect().size(); // Get the size of the available area
        if self.image.pixels.is_empty() || window_rezised(&size, &self.image.size) {
            self.image = render(&self.pool, size.x, size.y);
            println!("re-render")
        }

        let texture = ctx.load_texture(
            "dynamic_texture",
            self.image.clone(),
            egui::TextureOptions::NEAREST,
        );

        // Central Panel to display the green image
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                ui.image(&texture);
            });

        // Side Panel on the right side
        egui::SidePanel::right("right_panel")
            .resizable(false)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    if ui.button("render").clicked() {
                        self.image = render(&self.pool, size.x, size.y);
                        println!("re-render");
                    }

                    if ui
                        .checkbox(&mut self.accumulate, "Enable Feature")
                        .changed()
                    {
                        // Optionally handle changes to the checkbox state
                        println!("Checkbox state changed to: {}", self.accumulate);
                    }

                    let elapsed_milliseconds = start_time.elapsed().as_micros() as f32 / 1000.;
                    let time_string = format!("elapsed time in ms: {}", elapsed_milliseconds);

                    ui.label(time_string);
                });
            });
    }

    fn save(&mut self, _storage: &mut dyn eframe::Storage) {}

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {}

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // NOTE: a bright gray makes the shadows of the windows look weird.
        // We use a bit of transparency so that if the user switches on the
        // `transparent()` option they get immediate results.
        egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32()

        // _visuals.window_fill() would also be a natural choice
    }

    fn persist_egui_memory(&self) -> bool {
        true
    }

    fn raw_input_hook(&mut self, _ctx: &egui::Context, _raw_input: &mut egui::RawInput) {}
}

fn render(thread_pool: &ThreadPool, x_size: f32, y_size: f32) -> ColorImage {
    let pixel_count = (x_size * y_size) as usize;
    let mut pixels: Vec<u8> = Vec::with_capacity(pixel_count * 3);

    for y in 0..y_size as usize {
        for x in 0..x_size as usize {
            let x_color = x as f32 / x_size * 2. - 1.;
            let y_color = y as f32 / y_size * 2. - 1.;

            let pixel_position = glam::vec2(x_color, y_color);
            let pixel_color = per_pixel(pixel_position);

            let color = to_rgb(pixel_color);
            pixels.extend_from_slice(&color);
        }
    }

    egui::ColorImage::from_rgb([x_size as usize, y_size as usize], &pixels)
}

fn per_pixel(coords: glam::Vec2) -> glam::Vec3 {
    // (bx^2 + by^2)t^2 + 2*(axbx + ayby)t + (ax^2 + by^2 - r^2) = 0
    // where
    // a = ray origin
    // b = ray direction
    // r = sphere radius
    // t = hit distance

    let ray_origin: glam::Vec3 = glam::vec3(0., 0., -2.);
    let ray_direction: glam::Vec3 = glam::vec3(coords.x(), coords.y(), -1.);
    let radius: f32 = 0.5;

    let a: f32 = ray_direction.dot(ray_direction);
    let b: f32 = 2.0 * ray_direction.dot(ray_origin);
    let c: f32 = ray_origin.dot(ray_origin) - (radius * radius);

    // discriminant:
    // b^2 - 4*a*c
    let discriminant = b * b - 4. * a * c;

    if discriminant < 0. {
        return glam::vec3(0., 0., 0.);
    }

    // (-b +- discriminant) / 2a
    //let t0 = (-b + discriminant.sqrt()) / (2. * a);
    let closest_t = (-b - discriminant.sqrt()) / (2. * a);

    let hit_point: glam::Vec3 = ray_origin + ray_direction * closest_t;

    hit_point
}

fn window_rezised(current_size: &egui::Vec2, prev_size: &[usize; 2]) -> bool {
    current_size.x != prev_size[0] as f32 || current_size.y != prev_size[1] as f32
}

fn to_rgb(float_array: glam::Vec3) -> [u8; 3] {
    [
        (float_array.x() * 255.) as u8,
        (float_array.y() * 255.) as u8,
        (float_array.z() * 255.) as u8,
    ]
}
