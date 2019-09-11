//! Peak Automata
//! Visits the graph, traversing every node and assigning a new elevation. This algorithm takes a starting location
//! and elevation, and progressively lowers neighbors outwards from the starting location until a threshold is reached.
//!
//!
use crate::{
    dual_graph::{RegionEdge, RegionNode},
    HasElevation,
};
use nalgebra::{Point2, RealField};
use petgraph::{graph::NodeIndex, EdgeType, Graph};
use std::collections::HashSet;

#[derive(Default, Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PeakNode<T: RealField> {
    pub node: NodeIndex,
    pub elevation: T,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Settings<T: RealField> {
    peak_nodes: Vec<PeakNode<T>>,
    elevation: std::ops::Range<T>,
    radius: T,
    sharpness: T,
    step: T,
}
impl<T: RealField + From<f32>> Settings<T> {
    pub fn with_peak_nodes<'a, I>(mut self, soft_nodes: I) -> Self
    where
        T: 'a,
        I: IntoIterator<Item = PeakNode<T>>,
    {
        use std::iter::FromIterator;

        self.peak_nodes = Vec::from_iter(soft_nodes);

        self
    }

    pub fn with_radius(mut self, radius: T) -> Self {
        self.radius = radius;
        self
    }

    pub fn with_sharpness(mut self, sharpness: T) -> Self {
        self.sharpness = sharpness;
        self
    }

    pub fn with_step(mut self, step: T) -> Self {
        self.step = step;
        self
    }

    pub fn with_elevation(mut self, elevation: std::ops::Range<T>) -> Self {
        self.elevation = elevation;
        self
    }

    pub fn default() -> Self {
        Self {
            peak_nodes: Vec::default(),
            elevation: std::ops::Range {
                start: 1.0.into(),
                end: 0.0001.into(),
            },
            radius: 0.95.into(),
            sharpness: 0.2.into(),
            step: 0.1.into(),
        }
    }
}

fn fetch_or_err<T: Default, D: EdgeType>(
    graph: &Graph<RegionNode<T>, RegionEdge, D>,
    node: NodeIndex,
) -> Result<&RegionNode<T>, failure::Error> {
    Ok(graph
        .node_weight(node)
        .ok_or_else(|| failure::format_err!("Failed to fetch graph node: {:?}", node))?)
}
fn fetch_or_err_mut<T: Default, D: EdgeType>(
    graph: &mut Graph<RegionNode<T>, RegionEdge, D>,
    node: NodeIndex,
) -> Result<&mut RegionNode<T>, failure::Error> {
    Ok(graph
        .node_weight_mut(node)
        .ok_or_else(|| failure::format_err!("Failed to fetch graph node: {:?}", node))?)
}

pub fn visit<T, V, R, E>(
    region_graph: &mut Graph<RegionNode<V>, RegionEdge, E>,
    settings: &Settings<T>,
    rng: &mut R,
) -> Result<(), failure::Error>
where
    T: RealField,
    V: Default + HasElevation<T>,
    R: rand::Rng + ?Sized,
    E: EdgeType,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    for idx in &settings.peak_nodes {
        single_peak(region_graph, &settings, *idx, rng)?;
    }

    Ok(())
}

pub fn single_peak<T, V, R, E>(
    region_graph: &mut Graph<RegionNode<V>, RegionEdge, E>,
    settings: &Settings<T>,
    starting_node: PeakNode<T>,
    rng: &mut R,
) -> Result<(), failure::Error>
where
    T: RealField,
    V: Default + HasElevation<T>,
    R: rand::Rng + ?Sized,
    E: EdgeType,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    let mut completed = HashSet::with_capacity(region_graph.node_count());
    let mut queue = Vec::with_capacity(region_graph.node_count());

    let mut current_elevation = starting_node.elevation;
    fetch_or_err_mut(region_graph, starting_node.node)?
        .value
        .set_elevation(current_elevation);

    queue.push(starting_node.node);
    let mut i = 0;
    while i < queue.len() && current_elevation >= settings.elevation.end {
        let current_node = queue[i];
        let parent_elevation = fetch_or_err(region_graph, queue[i])?.value.elevation();
        current_elevation = parent_elevation * settings.radius;

        let mut walker = region_graph.neighbors(current_node).detach();
        while let Some(neighbor_idx) = walker.next(&region_graph) {
            if !completed.contains(&neighbor_idx.1) {
                let modifier = if settings.sharpness == num::zero() {
                    parent_elevation
                } else {
                    rng.gen::<T>() * settings.sharpness + settings.step - settings.sharpness
                };

                fetch_or_err_mut(region_graph, neighbor_idx.1)?
                    .value
                    .set_elevation(current_elevation + modifier);

                queue.push(neighbor_idx.1);
                completed.insert(neighbor_idx.1);
            }
        }
        i += 1;
    }

    Ok(())
}

pub fn node_for_coordinate<T, D>(
    graph: &Graph<RegionNode<T>, RegionEdge, D>,
    point: Point2<f32>,
) -> Option<NodeIndex>
where
    T: Default,
    D: EdgeType,
{
    use nalgebra::distance;
    use petgraph::visit::{IntoNodeReferences, NodeRef};

    graph.node_references().fold(None, |acc, region| {
        if let Some(last) = acc {
            if distance(&graph.node_weight(last).expect("Bad graph").pos, &point)
                < distance(&region.weight().pos, &point)
            {
                Some(last)
            } else {
                Some(region.id())
            }
        } else {
            Some(region.id())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual_graph::{gen_dual_graph, BorderGraph};
    use imageproc::drawing::Point as ImgPoint;
    use nalgebra::Vector2;
    use petgraph::visit::IntoNodeReferences;
    use rand::Rng;
    use rand::SeedableRng;
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
    pub fn island_visitor() {
        let dims = Vector2::new(1024.0, 1024.0);
        let mut imgbuf = image::ImageBuffer::from_pixel(
            dims.x as u32,
            dims.y as u32,
            image::Rgb([222, 222, 222]),
        );

        let mut rng =
            rand_xorshift::XorShiftRng::from_seed([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);

        let (mut region_graph, border_graph) =
            gen_dual_graph::<TestInner, (), rand_xorshift::XorShiftRng>(dims, 500, 2, &mut rng);

        // Start at the center
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

        visit(&mut region_graph, &settings, &mut rng).unwrap();

        draw_island(
            &mut imgbuf,
            &region_graph,
            &border_graph,
            |region, border_graph| {
                //let color = region.value.height as i32;
                let elevation = region.value.elevation();
                let color = if elevation > 0.3 {
                    (255.0 / (1.0 - elevation)) as u8
                } else {
                    0
                };

                (
                    image::Rgb([color, color, color]),
                    region
                        .borders
                        .iter()
                        .filter_map(|idx| {
                            let node = border_graph.node_weight(*idx).expect("Bad graphs");

                            Some(ImgPoint::<i32>::new(node.pos.x as i32, node.pos.y as i32))
                        })
                        .collect(),
                )
            },
        );

        imgbuf.save("output/island.png").unwrap();
    }

    pub(crate) fn draw_island<
        RG: IntoNodeReferences,
        N: Fn(
            &<RG as petgraph::visit::Data>::NodeWeight,
            &BorderGraph,
        ) -> (<I as image::GenericImageView>::Pixel, Vec<ImgPoint<i32>>),
        I,
    >(
        imgbuf: &mut I,
        region_graph: RG,
        border_graph: &BorderGraph,
        node_color: N,
    ) where
        I: image::GenericImage,
        I::Pixel: 'static,
        <<I as image::GenericImageView>::Pixel as image::Pixel>::Subpixel:
            conv::ValueInto<f32> + imageproc::definitions::Clamp<f32>,
    {
        use petgraph::visit::NodeRef;
        for node in region_graph.node_references() {
            let (color, points) = node_color(node.weight(), border_graph);
            let len = points.len();
            let dedup = points
                .into_iter()
                .fold(Vec::with_capacity(len), |mut acc, point| {
                    if !acc.contains(&point) {
                        acc.push(point);
                    }
                    acc
                });

            if dedup.len() >= 3 {
                imageproc::drawing::draw_convex_polygon_mut(imgbuf, dedup.as_slice(), color);
            }
        }
    }
}

#[cfg(test)]
mod rbf_interp_tests {
    use super::*;
    use crate::dual_graph::{gen_dual_graph, BorderGraph};
    use crate::HasValue;
    use nalgebra::Vector2;
    use petgraph::visit::{IntoNodeReferences, NodeCount, NodeRef};
    use rand::Rng;
    use rand::SeedableRng;

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
    pub fn rbf_interp() {
        let dims = Vector2::new(1024.0, 1024.0);
        let mut imgbuf = image::ImageBuffer::from_pixel(
            dims.x as u32,
            dims.y as u32,
            image::Rgb([222, 222, 222]),
        );

        let mut rng =
            rand_xorshift::XorShiftRng::from_seed([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);

        let (mut region_graph, border_graph) =
            gen_dual_graph::<TestInner, (), rand_xorshift::XorShiftRng>(dims, 500, 2, &mut rng);

        // Start at the center
        let center = Point2::from(dims / 2.0);
        let soft_points = (0..10)
            .map(|_| {
                let height = rng.gen_range(0.5, 0.8);
                let x = rng.gen_range(dims.x / 3.0, dims.x - (dims.x / 3.0));
                let y = rng.gen_range(dims.y / 3.0, dims.y - (dims.y / 3.0));

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

        visit(&mut region_graph, &settings, &mut rng).unwrap();

        draw_island(
            &mut imgbuf,
            &region_graph,
            &border_graph,
            |region, border_graph| {
                region
                    .borders
                    .iter()
                    .filter_map(|idx| {
                        let node = border_graph.node_weight(*idx).expect("Bad graphs");
                        Some(Point2::<i32>::new(node.pos.x as i32, node.pos.y as i32))
                    })
                    .collect()
            },
            |region, graph, point| {
                use rbf_interp::{DistanceFunction, PtValue, Rbf};
                use smallvec::SmallVec;

                let mut points = SmallVec::<[PtValue<f32>; 32]>::default();
                graph.neighbors_undirected(region.0).for_each(|idx| {
                    let node = graph.node_weight(idx).unwrap();
                    points.push(PtValue::new(node.pos.x, node.pos.y, node.value.elevation()));
                });
                points.push(PtValue::new(
                    region.1.pos.x,
                    region.1.pos.y,
                    region.1.value().elevation(),
                ));

                let height = Rbf::new(&points, DistanceFunction::Linear, None)
                    .interp_point((point.x as f32, point.y as f32));
                let color = (255.0 * height.min(1.0).max(0.0)) as u8;

                image::Rgb([color, color, color])
            },
        );

        imgbuf.save("output/rbf_interp.png").unwrap();
    }

    fn draw_island<
        RG: IntoNodeReferences + NodeCount,
        N: Fn(&<RG as petgraph::visit::Data>::NodeWeight, &BorderGraph) -> Vec<Point2<i32>>,
        P: (Fn(
                <RG as IntoNodeReferences>::NodeRef,
                &RG,
                &Point2<i32>,
            ) -> <I as image::GenericImageView>::Pixel)
            + Clone,
        I,
    >(
        imgbuf: &mut I,
        region_graph: RG,
        border_graph: &BorderGraph,
        node_points: N,
        point_color: P,
    ) where
        <RG as petgraph::visit::Data>::NodeWeight: HasValue,
        <<RG as petgraph::visit::Data>::NodeWeight as HasValue>::Value: HasElevation<f32>,
        <RG as IntoNodeReferences>::NodeRef: Send,
        I: image::GenericImage,
        I::Pixel: 'static,
        <<I as image::GenericImageView>::Pixel as image::Pixel>::Subpixel:
            conv::ValueInto<f32> + imageproc::definitions::Clamp<f32>,
    {
        region_graph
            .node_references()
            .enumerate()
            .for_each(|(i, node)| {
                let points = node_points(node.weight(), border_graph);
                let len = points.len();
                let dedup = points
                    .into_iter()
                    .fold(Vec::with_capacity(len), |mut acc, point| {
                        if !acc.contains(&point) {
                            acc.push(point);
                        }
                        acc
                    });

                if dedup.len() >= 3 {
                    draw_convex_polygon_mut(
                        imgbuf,
                        dedup.as_slice(),
                        &region_graph,
                        point_color.clone(),
                        node,
                    );
                }
            });
    }

    fn draw_convex_polygon_mut<
        C,
        P: (Fn(<RG as IntoNodeReferences>::NodeRef, &RG, &Point2<i32>) -> C::Pixel) + Clone,
        RG: IntoNodeReferences + NodeCount,
    >(
        canvas: &mut C,
        poly: &[Point2<i32>],
        graph: &RG,
        point_color: P,
        region: <RG as IntoNodeReferences>::NodeRef,
    ) where
        C: imageproc::drawing::Canvas,
        C::Pixel: 'static,
    {
        use std::cmp::{max, min};

        if poly.is_empty() {
            return;
        }
        if poly[0] == poly[poly.len() - 1] {
            panic!(
                "First point {:?} == last point {:?}",
                poly[0],
                poly[poly.len() - 1]
            );
        }

        let mut y_min = std::i32::MAX;
        let mut y_max = std::i32::MIN;
        for p in poly {
            y_min = min(y_min, p.y);
            y_max = max(y_max, p.y);
        }

        let (width, height) = canvas.dimensions();

        // Intersect polygon vertical range with image bounds
        y_min = max(0, min(y_min, height as i32 - 1));
        y_max = max(0, min(y_max, height as i32 - 1));

        // TODO: This avoids a bug in the voronoi implementation causing overlaping entries

        let mut closed = Vec::with_capacity(poly.len() + 1);
        for p in poly {
            closed.push(*p);
        }
        closed.push(poly[0]);

        let edges: Vec<&[Point2<i32>]> = closed.windows(2).collect();
        let mut intersections: Vec<i32> = Vec::new();

        for y in y_min..y_max + 1 {
            for edge in &edges {
                let p0 = edge[0];
                let p1 = edge[1];

                if p0.y <= y && p1.y >= y || p1.y <= y && p0.y >= y {
                    // Need to handle horizontal lines specially
                    if p0.y == p1.y {
                        intersections.push(p0.x);
                        intersections.push(p1.x);
                    } else {
                        let fraction = (y - p0.y) as f32 / (p1.y - p0.y) as f32;
                        let inter = p0.x as f32 + fraction * (p1.x - p0.x) as f32;
                        intersections.push(inter.round() as i32);
                    }
                }
            }

            intersections.sort();
            let mut i = 0;
            loop {
                // Handle points where multiple lines intersect
                while i + 1 < intersections.len() && intersections[i] == intersections[i + 1] {
                    i += 1;
                }
                if i >= intersections.len() {
                    break;
                }
                if i + 1 == intersections.len() {
                    let color = point_color(region, graph, &Point2::new(intersections[i], y));
                    draw_if_in_bounds(canvas, intersections[i], y, color);
                    break;
                }
                let from = max(0, min(intersections[i], width as i32 - 1));
                let to = max(0, min(intersections[i + 1], width as i32 - 1));
                for x in from..to + 1 {
                    let color = point_color(region, graph, &Point2::new(x, y));
                    canvas.draw_pixel(x as u32, y as u32, color);
                }
                i += 2;
            }

            intersections.clear();
        }

        for edge in &edges {
            let start = (edge[0].x as f32, edge[0].y as f32);
            let end = (edge[1].x as f32, edge[1].y as f32);
            draw_line_segment_mut(canvas, start, end, graph, region, point_color.clone());
        }
    }

    fn draw_if_in_bounds<C>(canvas: &mut C, x: i32, y: i32, color: C::Pixel)
    where
        C: imageproc::drawing::Canvas,
        C::Pixel: 'static,
    {
        if x >= 0 && x < canvas.width() as i32 && y >= 0 && y < canvas.height() as i32 {
            canvas.draw_pixel(x as u32, y as u32, color);
        }
    }

    fn draw_line_segment_mut<C, P, RG>(
        canvas: &mut C,
        start: (f32, f32),
        end: (f32, f32),
        graph: &RG,
        region: <RG as IntoNodeReferences>::NodeRef,
        point_color: P,
    ) where
        RG: IntoNodeReferences + NodeCount,
        P: (Fn(<RG as IntoNodeReferences>::NodeRef, &RG, &Point2<i32>) -> C::Pixel) + Clone,
        C: imageproc::drawing::Canvas,
        C::Pixel: 'static,
    {
        let (width, height) = canvas.dimensions();
        let in_bounds = |x, y| x >= 0 && x < width as i32 && y >= 0 && y < height as i32;

        let line_iterator = imageproc::drawing::BresenhamLineIter::new(start, end);

        for point in line_iterator {
            let x = point.0;
            let y = point.1;

            if in_bounds(x, y) {
                let color = point_color(region, &graph, &Point2::new(x, y));
                canvas.draw_pixel(x as u32, y as u32, color);
            }
        }
    }
}

#[cfg(test)]
mod spade_tests {
    use super::*;
    use crate::dual_graph::{gen_dual_graph, BorderGraph};
    use nalgebra::Vector2;
    use petgraph::visit::{IntoNodeReferences, NodeRef};
    use rand::Rng;
    use rand::SeedableRng;
    use spade::delaunay::*;

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

    impl spade::HasPosition for &RegionNode<TestInner> {
        type Point = Point2<f32>;

        fn position(&self) -> Point2<f32> {
            self.pos
        }
    }

    #[test]
    fn dt_spade_interp() {
        let dims = Vector2::new(1024.0, 1024.0);
        let mut imgbuf = image::ImageBuffer::from_pixel(
            dims.x as u32,
            dims.y as u32,
            image::Rgb([222, 222, 222]),
        );

        let mut rng =
            rand_xorshift::XorShiftRng::from_seed([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);

        let (mut region_graph, border_graph) =
            gen_dual_graph::<TestInner, (), rand_xorshift::XorShiftRng>(dims, 500, 2, &mut rng);

        // Start at the center
        let center = Point2::from(dims / 2.0);
        let soft_points = (0..10)
            .map(|_| {
                let height = rng.gen_range(0.5, 0.8);
                let x = rng.gen_range(dims.x / 3.0, dims.x - (dims.x / 3.0));
                let y = rng.gen_range(dims.y / 3.0, dims.y - (dims.y / 3.0));
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

        visit(&mut region_graph, &settings, &mut rng).unwrap();

        // Try building the DT out of the graph

        let mut dt = DelaunayTriangulation::<
            &RegionNode<TestInner>,
            spade::kernels::FloatKernel,
            DelaunayWalkLocate,
        >::default();
        for node in region_graph.node_references() {
            dt.insert(node.1);
        }

        draw_island(
            &mut imgbuf,
            &region_graph,
            &border_graph,
            &dt,
            |region, border_graph| {
                region
                    .borders
                    .iter()
                    .filter_map(|idx| {
                        let node = border_graph.node_weight(*idx).expect("Bad graphs");

                        Some(Point2::<i32>::new(node.pos.x as i32, node.pos.y as i32))
                    })
                    .collect()
            },
            |dt, point| {
                let p = Point2::new(point.x as f32, point.y as f32);
                let height = dt.nn_interpolation(&p, |v| v.value.elevation()).unwrap();
                let color = (255.0 * height.min(1.0).max(0.0)) as u8;
                image::Rgb([color, color, color])
            },
        );

        imgbuf.save("output/dt_spade_interp.png").unwrap();
    }

    fn draw_island<
        RG: IntoNodeReferences,
        N: Fn(&<RG as petgraph::visit::Data>::NodeWeight, &BorderGraph) -> Vec<Point2<i32>>,
        P: (Fn(
                &DelaunayTriangulation<
                    &RegionNode<TestInner>,
                    spade::kernels::FloatKernel,
                    DelaunayWalkLocate,
                >,
                &Point2<i32>,
            ) -> <I as image::GenericImageView>::Pixel)
            + Clone,
        I,
    >(
        imgbuf: &mut I,
        region_graph: RG,
        border_graph: &BorderGraph,
        dt: &DelaunayTriangulation<
            &RegionNode<TestInner>,
            spade::kernels::FloatKernel,
            DelaunayWalkLocate,
        >,
        node_points: N,
        point_color: P,
    ) where
        I: image::GenericImage,
        I::Pixel: 'static,
        <<I as image::GenericImageView>::Pixel as image::Pixel>::Subpixel:
            conv::ValueInto<f32> + imageproc::definitions::Clamp<f32>,
    {
        for node in region_graph.node_references() {
            let points = node_points(node.weight(), border_graph);
            let len = points.len();
            let dedup = points
                .into_iter()
                .fold(Vec::with_capacity(len), |mut acc, point| {
                    if !acc.contains(&point) {
                        acc.push(point);
                    }
                    acc
                });

            if dedup.len() >= 3 {
                draw_convex_polygon_mut(imgbuf, dedup.as_slice(), &dt, point_color.clone());
            }
        }
    }

    fn draw_convex_polygon_mut<
        C,
        P: (Fn(
                &DelaunayTriangulation<
                    &RegionNode<TestInner>,
                    spade::kernels::FloatKernel,
                    DelaunayWalkLocate,
                >,
                &Point2<i32>,
            ) -> C::Pixel)
            + Clone,
    >(
        canvas: &mut C,
        poly: &[Point2<i32>],
        dt: &DelaunayTriangulation<
            &RegionNode<TestInner>,
            spade::kernels::FloatKernel,
            DelaunayWalkLocate,
        >,
        point_color: P,
    ) where
        C: imageproc::drawing::Canvas,
        C::Pixel: 'static,
    {
        use std::cmp::{max, min};

        if poly.is_empty() {
            return;
        }
        if poly[0] == poly[poly.len() - 1] {
            panic!(
                "First point {:?} == last point {:?}",
                poly[0],
                poly[poly.len() - 1]
            );
        }

        let mut y_min = std::i32::MAX;
        let mut y_max = std::i32::MIN;
        for p in poly {
            y_min = min(y_min, p.y);
            y_max = max(y_max, p.y);
        }

        let (width, height) = canvas.dimensions();

        // Intersect polygon vertical range with image bounds
        y_min = max(0, min(y_min, height as i32 - 1));
        y_max = max(0, min(y_max, height as i32 - 1));

        let mut closed = Vec::with_capacity(poly.len() + 1);
        for p in poly {
            closed.push(*p);
        }
        closed.push(poly[0]);

        let edges: Vec<&[Point2<i32>]> = closed.windows(2).collect();
        let mut intersections: Vec<i32> = Vec::new();

        for y in y_min..y_max + 1 {
            for edge in &edges {
                let p0 = edge[0];
                let p1 = edge[1];

                if p0.y <= y && p1.y >= y || p1.y <= y && p0.y >= y {
                    // Need to handle horizontal lines specially
                    if p0.y == p1.y {
                        intersections.push(p0.x);
                        intersections.push(p1.x);
                    } else {
                        let fraction = (y - p0.y) as f32 / (p1.y - p0.y) as f32;
                        let inter = p0.x as f32 + fraction * (p1.x - p0.x) as f32;
                        intersections.push(inter.round() as i32);
                    }
                }
            }

            intersections.sort();
            let mut i = 0;
            loop {
                // Handle points where multiple lines intersect
                while i + 1 < intersections.len() && intersections[i] == intersections[i + 1] {
                    i += 1;
                }
                if i >= intersections.len() {
                    break;
                }
                if i + 1 == intersections.len() {
                    let color = point_color(dt, &Point2::new(intersections[i], y));
                    draw_if_in_bounds(canvas, intersections[i], y, color);
                    break;
                }
                let from = max(0, min(intersections[i], width as i32 - 1));
                let to = max(0, min(intersections[i + 1], width as i32 - 1));
                for x in from..to + 1 {
                    let color = point_color(dt, &Point2::new(x, y));
                    canvas.draw_pixel(x as u32, y as u32, color);
                }
                i += 2;
            }

            intersections.clear();
        }

        for edge in &edges {
            let start = (edge[0].x as f32, edge[0].y as f32);
            let end = (edge[1].x as f32, edge[1].y as f32);
            draw_line_segment_mut(canvas, start, end, &dt, point_color.clone());
        }
    }

    fn draw_if_in_bounds<C>(canvas: &mut C, x: i32, y: i32, color: C::Pixel)
    where
        C: imageproc::drawing::Canvas,
        C::Pixel: 'static,
    {
        if x >= 0 && x < canvas.width() as i32 && y >= 0 && y < canvas.height() as i32 {
            canvas.draw_pixel(x as u32, y as u32, color);
        }
    }

    fn draw_line_segment_mut<C, P>(
        canvas: &mut C,
        start: (f32, f32),
        end: (f32, f32),
        dt: &DelaunayTriangulation<
            &RegionNode<TestInner>,
            spade::kernels::FloatKernel,
            DelaunayWalkLocate,
        >,
        point_color: P,
    ) where
        P: (Fn(
                &DelaunayTriangulation<
                    &RegionNode<TestInner>,
                    spade::kernels::FloatKernel,
                    DelaunayWalkLocate,
                >,
                &Point2<i32>,
            ) -> C::Pixel)
            + Clone,
        C: imageproc::drawing::Canvas,
        C::Pixel: 'static,
    {
        let (width, height) = canvas.dimensions();
        let in_bounds = |x, y| x >= 0 && x < width as i32 && y >= 0 && y < height as i32;

        let line_iterator = imageproc::drawing::BresenhamLineIter::new(start, end);

        for point in line_iterator {
            let x = point.0;
            let y = point.1;

            if in_bounds(x, y) {
                let color = point_color(dt, &Point2::new(x, y));
                canvas.draw_pixel(x as u32, y as u32, color);
            }
        }
    }
}
