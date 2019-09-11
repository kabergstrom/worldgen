use crate::{
    dual_graph::{RegionEdge, RegionNode},
    HasElevation, HasTemperature, HasWind,
};
use nalgebra::{Point2, RealField, Vector2};
use petgraph::{EdgeType, Graph};

pub struct Settings<T: RealField> {
    simulation_duration: f32, // 1 unit = 1 day
    start_speed: T,
    start_from: Vector2<T>,
}
impl<T: RealField + From<f32>> Settings<T> {
    pub fn default() -> Self {
        Self {
            simulation_duration: 365.0.into(),
            start_speed: 5.0.into(),
            start_from: Vector2::new(1.0.into(), 0.0.into()),
        }
    }
}

pub fn visit<T, V, R, E>(
    _region_graph: &mut Graph<RegionNode<V>, RegionEdge, E>,
    _settings: &Settings<T>,
    _rng: &mut R,
) -> Result<(), failure::Error>
where
    T: RealField,
    V: Default + HasElevation<T> + HasWind<T> + HasTemperature<T>,
    R: rand::Rng + ?Sized,
    E: EdgeType,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    // Just place wind direction on all elements for testing drawing

    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::peak_automata;
    use crate::{
        dual_graph::{gen_dual_graph, RegionGraph},
        HasValue,
    };
    use petgraph::visit::{IntoNodeReferences, NodeCount};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    struct TestInner {
        elevation: f32,
        temperature: f32,
        wind: Vector2<f32>,
    }
    impl Default for TestInner {
        fn default() -> Self {
            Self {
                elevation: 0.0,
                temperature: 21.0,
                wind: Vector2::new(0.0, 0.0),
            }
        }
    }

    impl HasElevation<f32> for TestInner {
        fn elevation(&self) -> f32 {
            self.elevation
        }
        fn set_elevation(&mut self, height: f32) {
            self.elevation = height;
        }
    }
    impl HasTemperature<f32> for TestInner {
        fn temperature(&self) -> f32 {
            self.temperature
        }
        fn set_temperature(&mut self, temp: f32) {
            self.temperature = temp;
        }
    }

    impl HasWind<f32> for TestInner {
        fn wind_vector(&self) -> Vector2<f32> {
            self.wind
        }
        fn set_wind_vector(&mut self, wind: Vector2<f32>) {
            self.wind = wind;
        }
    }

    fn apply_peak_automata<R>(
        dims: &Vector2<f32>,
        region_graph: &mut RegionGraph<TestInner>,
        rng: &mut R,
    ) where
        R: rand::Rng,
    {
        use crate::peak_automata::{node_for_coordinate, PeakNode, Settings};

        let center = Point2::from(dims / 2.0);
        let soft_points = (0..10)
            .map(|_| {
                let height = rng.gen_range(0.3, 0.8);
                let x = rng.gen_range(-500.0, 500.0);
                let y = rng.gen_range(-500.0, 500.0);
                PeakNode {
                    node: node_for_coordinate(
                        &region_graph,
                        Point2::new(center.x + x, center.y + y),
                    )
                    .expect("wut"),
                    elevation: height,
                }
            })
            .collect::<Vec<_>>();

        let settings = Settings::<f32>::default().with_peak_nodes(soft_points);

        peak_automata::visit(region_graph, &settings, rng).unwrap();
    }

    #[test]
    fn wind_test() {
        use petgraph::visit::NodeRef;

        let dims = Vector2::new(1024.0, 1024.0);
        let mut imgbuf =
            image::ImageBuffer::from_pixel(dims.x as u32, dims.y as u32, image::Rgb([0, 0, 0]));

        let mut rng = XorShiftRng::from_seed([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);

        let (mut region_graph, border_graph) =
            gen_dual_graph::<TestInner, (), XorShiftRng>(dims, 8000, 2, &mut rng);

        apply_peak_automata(&dims, &mut region_graph, &mut rng);

        let settings = Settings::<f32>::default();
        visit(&mut region_graph, &settings, &mut rng).unwrap();

        for node in region_graph.node_references() {
            let region = node.weight();
            let value = region.value();

            if value.wind_vector().magnitude() > 0.0 {
                let start = &region.pos;
                let end = start + (value.wind_vector().normalize() * 10.0);

                imageproc::drawing::draw_antialiased_line_segment_mut(
                    &mut imgbuf,
                    (start.x as i32, start.y as i32),
                    (end.x as i32, end.y as i32),
                    image::Rgb([0, 222, 222]),
                    imageproc::pixelops::interpolate,
                );

                // Draw a circle at the source direction
                imageproc::drawing::draw_filled_circle_mut(
                    &mut imgbuf,
                    (start.x as i32, start.y as i32),
                    2,
                    image::Rgb([0, 222, 222]),
                );
            }
        }

        imgbuf.save("output/simple_wind.png").unwrap();
    }
}
