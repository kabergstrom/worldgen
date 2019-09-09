use crate::{
    dual_graph::{RegionEdge, RegionNode},
    HasElevation, HasWind,
};
use nalgebra::{Point2, RealField};
use petgraph::{visit::IntoNodeReferences, EdgeType, Graph};
use rbf_interp::{DistanceFunction, PtValue, Rbf};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual_graph::{gen_dual_graph, BorderGraph};
    use imageproc::drawing::Point as ImgPoint;
    use nalgebra::{Point2, Vector2};
    use rand::Rng;
    use rand::SeedableRng;

    use crate::peak_automata::{node_for_coordinate, visit, PeakNode, Settings};

    #[derive(Default)]
    struct TestInner {
        elevation: f32,
    }
    impl HasElevation<f32> for TestInner {
        fn elevation(&self) -> f32 {
            self.elevation
        }
        fn set_elevation(&mut self, height: f32) {
            self.elevation = height;
        }
    }

    #[test]
    pub fn rbf() {
        let dims = Vector2::new(1024.0, 1024.0);
        let mut imgbuf = image::ImageBuffer::from_pixel(
            dims.x as u32,
            dims.y as u32,
            image::Rgb([222, 222, 222]),
        );
        let (mut region_graph, mut border_graph) = gen_dual_graph::<TestInner, ()>(dims, 8000, 2);

        let mut rng =
            rand_xorshift::XorShiftRng::from_seed([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);

        // Start at the center
        let center = Point2::from(dims / 2.0);
        let center_node = node_for_coordinate(&region_graph, center).expect("wut");
        let soft_points = (0..10)
            .map(|i| {
                let height = rng.gen_range(0.5, 0.8);
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

        let settings = Settings::<f32>::default_f32().with_peak_nodes(soft_points);

        visit(&mut region_graph, &settings, &mut rng).unwrap();

        let mut points = Vec::with_capacity(region_graph.node_count());
        region_graph.node_references().for_each(|node| {
            points.push(PtValue::new(
                node.1.pos.x,
                node.1.pos.y,
                node.1.value.elevation(),
            ));
        });

        let rbf = Rbf::new(&points, DistanceFunction::Linear, None);

        imgbuf.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
            let v = (255.0 * rbf.interp_point((x as f32, y as f32))) as u8;
            *pixel = image::Rgb([v, 0, 0])
        });

        imgbuf.save("output/rbf.png").unwrap();
    }
}
