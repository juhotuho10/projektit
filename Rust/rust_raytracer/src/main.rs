mod camera;

use eframe::egui;
use egui::pos2;
use egui::ColorImage;
use egui::Frame;
use egui::Key;
use egui::TextureHandle;
use glam::{vec3a, Vec3A};
use rand::{thread_rng, Rng};
use std::time::Instant;

use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 1000.0]),
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

        if self.image.pixels.is_empty() || window_rezised(&size, &self.image.size) || self.moved {
            self.image = render(&self.pool, size.x, size.y);
            println!("re-render");
            self.moved = false;
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

                if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
                    // `interact_pos` gives you the position of the mouse when it's over an interactive area
                    println!("Mouse Position: ({:.2}, {:.2})", pos.x, pos.y);
                } else {
                    println!("Mouse is not over an interactive area");
                }
            });

        let transparent_frame =
            Frame::none().fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, 100));
        // Side Panel on the right side

        if ctx.input(|i: &egui::InputState| i.pointer.secondary_down()) {
            egui::Context::send_viewport_cmd(
                ctx,
                egui::ViewportCommand::CursorGrab(egui::CursorGrab::Confined),
            );

            egui::Context::send_viewport_cmd(ctx, egui::ViewportCommand::CursorVisible(false));
            egui::Context::send_viewport_cmd(
                ctx,
                egui::ViewportCommand::CursorPosition(pos2(size.x / 2., size.y / 2.)),
            );
        } else {
            egui::Context::send_viewport_cmd(
                ctx,
                egui::ViewportCommand::CursorGrab(egui::CursorGrab::None),
            );
            egui::Context::send_viewport_cmd(ctx, egui::ViewportCommand::CursorVisible(true));
        }

        egui::SidePanel::right("right_panel")
            .resizable(false)
            .frame(transparent_frame)
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
        let y = y as f32 / y_size * 2. - 1.;
        for x in 0..x_size as usize {
            let x = x as f32 / x_size * 2. - 1.;

            let pixel_color = per_pixel(x, y);

            let color: [u8; 3] = to_rgb(pixel_color);
            pixels.extend_from_slice(&color);
        }
    }

    egui::ColorImage::from_rgb([x_size as usize, y_size as usize], &pixels)
}

fn per_pixel(x: f32, y: f32) -> Vec3A {
    // (bx^2 + by^2)t^2 + 2*(axbx + ayby)t + (ax^2 + by^2 - r^2) = 0
    // where
    // a = ray origin
    // b = ray direction
    // r = sphere radius
    // t = hit distance

    let ray_origin = vec3a(0., 0., -1.);
    let ray_direction = vec3a(x, y, -1.);
    let sphere_origin = vec3a(0., 0., 0.);
    let light_direction = vec3a(-1., -1., -1.).normalize();
    let radius: f32 = 0.5;

    let a: f32 = ray_direction.dot(ray_direction);
    let b: f32 = 2.0 * ray_direction.dot(ray_origin);
    let c: f32 = ray_origin.dot(ray_origin) - (radius * radius);

    // discriminant:
    // b^2 - 4*a*c
    let discriminant = b * b - 4. * a * c;

    if discriminant < 0. {
        // we missed the sphere
        return Vec3A::splat(0.);
    }

    // (-b +- discriminant) / 2a
    //let t0 = (-b + discriminant.sqrt()) / (2. * a);
    let closest_t = (-b - discriminant.sqrt()) / (2. * a);

    let hit_point = ray_origin + ray_direction * closest_t;

    let sphere_normal = (hit_point - sphere_origin).normalize();

    // cosine of the angle between hitpoin and the light direction
    // min light intenstiy is 0
    let light_intensity = sphere_normal.dot(-light_direction).max(0.);

    vec3a(1., 0., 1.) * light_intensity
}

fn window_rezised(current_size: &egui::Vec2, prev_size: &[usize; 2]) -> bool {
    current_size.x != prev_size[0] as f32 || current_size.y != prev_size[1] as f32
}

fn to_rgb(float_array: Vec3A) -> [u8; 3] {
    [
        (float_array.x * 255.) as u8,
        (float_array.y * 255.) as u8,
        (float_array.z * 255.) as u8,
    ]
}
