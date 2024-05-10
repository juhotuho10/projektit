struct Camera {
    pub position: glam::Vec3,
    pub direction: glam::Vec3,
    pub ray_direction: Vec<glam::Vec3>,
    near_clip: f32,
    far_clip: f32,
    vertical_fov: f32,
    speed: f32,
}

impl Camera {
    fn new() -> Camera {
        Camera {
            position: glam::vec3(0., 0., 3.),
            direction: glam::vec3(0., 0., -1.),
            ray_direction: vec![],
            near_clip: 0.1,
            far_clip: 100.0,
            vertical_fov: 45.0,
            speed: 0.1,
        }
    }

    fn on_update(&self, timestep: f32) {}

    fn foward(&mut self) {
        self.position += self.direction.normalize() * self.speed;
    }

    fn backward(&mut self) {
        self.position -= self.direction.normalize() * self.speed;
    }
    fn left(&mut self) {
        self.position += glam::vec3(0., self.speed, 0.)
    }
    fn right(&mut self) {
        self.position -= glam::vec3(0., self.speed, 0.)
    }
}
