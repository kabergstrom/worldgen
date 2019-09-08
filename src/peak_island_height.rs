//! Peak Island Height
//! Visits the graph, traversing every node and assigning a new elevation. This algorithm takes a starting location
//! and elevation, and progressively lowers neighbors outwards from the starting location until a threshold is reached.
//!
//!
use crate::{
    dual_graph::{BorderEdge, BorderGraph, BorderNode, RegionEdge, RegionNode},
    HasElevation,
};
use nalgebra::Point2;
use petgraph::{
    graph::{IndexType, NodeIndex},
    EdgeType, Graph,
};
use std::collections::HashSet;

use num::Zero;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Settings<T, Ix: IndexType> {
    starting_nodes: Vec<NodeIndex<Ix>>,
    starting_elevation: T,
    end_elevation: T,
    radius: T,
    sharpness: T,
    step: T,
}

fn default_settings_f32<Ix: IndexType>(starting_nodes: Vec<NodeIndex<Ix>>) -> Settings<f32, Ix> {
    Settings {
        starting_nodes,
        starting_elevation: 1.0,
        radius: 0.95,
        end_elevation: 0.0001,
        sharpness: 0.2,
        step: 0.1,
    }
}
fn default_settings_f64<Ix: IndexType>(starting_nodes: Vec<NodeIndex<Ix>>) -> Settings<f64, Ix> {
    Settings {
        starting_nodes,
        starting_elevation: 1.0,
        end_elevation: 0.0001,
        radius: 0.95,
        sharpness: 0.2,
        step: 0.1,
    }
}

fn fetch_or_err<T: Default, D: EdgeType, Ix: IndexType>(
    graph: &Graph<RegionNode<T>, RegionEdge, D, Ix>,
    node: NodeIndex<Ix>,
) -> Result<&RegionNode<T>, failure::Error> {
    Ok(graph
        .node_weight(node)
        .ok_or_else(|| failure::format_err!("Failed to fetch graph node: {:?}", node))?)
}
fn fetch_or_err_mut<T: Default, D: EdgeType, Ix: IndexType>(
    graph: &mut Graph<RegionNode<T>, RegionEdge, D, Ix>,
    node: NodeIndex<Ix>,
) -> Result<&mut RegionNode<T>, failure::Error> {
    Ok(graph
        .node_weight_mut(node)
        .ok_or_else(|| failure::format_err!("Failed to fetch graph node: {:?}", node))?)
}

pub fn visit<RT, BT, R, H, D, Ix>(
    region_graph: &mut Graph<RegionNode<RT>, RegionEdge, D, Ix>,
    border_graph: &mut Graph<BorderNode<BT>, BorderEdge, D, Ix>,
    settings: &Settings<H, Ix>,
    rng: &mut R,
) -> Result<(), failure::Error>
where
    H: PartialOrd
        + PartialEq
        + Add<Output = H>
        + Mul<Output = H>
        + Sub<Output = H>
        + Zero
        + Copy
        + Sized
        + std::fmt::Debug,
    RT: Default + HasElevation<H>,
    R: rand::Rng + ?Sized,
    D: EdgeType,
    Ix: IndexType,
    rand::distributions::Standard: rand::distributions::Distribution<H>,
{
    for node in settings.starting_nodes.iter() {
        single_peak(region_graph, border_graph, &settings, *node, rng)?;
    }

    Ok(())
}

pub fn single_peak<RT, BT, R, H, D, Ix>(
    region_graph: &mut Graph<RegionNode<RT>, RegionEdge, D, Ix>,
    border_graph: &mut Graph<BorderNode<BT>, BorderEdge, D, Ix>,
    settings: &Settings<H, Ix>,
    starting_node: NodeIndex<Ix>,
    rng: &mut R,
) -> Result<(), failure::Error>
where
    H: PartialOrd
        + PartialEq
        + Add<Output = H>
        + Mul<Output = H>
        + Sub<Output = H>
        + Zero
        + Copy
        + Sized
        + std::fmt::Debug,
    RT: Default + HasElevation<H>,
    R: rand::Rng + ?Sized,
    D: EdgeType,
    Ix: IndexType,
    rand::distributions::Standard: rand::distributions::Distribution<H>,
{
    let mut completed = HashSet::with_capacity(region_graph.node_count());
    let mut queue = Vec::with_capacity(region_graph.node_count());

    let mut current_elevation = settings.starting_elevation;
    fetch_or_err_mut(region_graph, starting_node)?
        .value
        .set_elevation(current_elevation);

    queue.push(starting_node);
    let mut i = 0;
    while i < queue.len() && current_elevation >= settings.end_elevation {
        let current_node = queue[i];
        let parent_elevation = fetch_or_err(region_graph, queue[i])?.value.elevation();
        current_elevation = parent_elevation * settings.radius;

        let mut walker = region_graph.neighbors(current_node).detach();
        while let Some(neighbor_idx) = walker.next(&region_graph) {
            if !completed.contains(&neighbor_idx.1) {
                let modifier = if settings.sharpness == num::zero() {
                    parent_elevation
                } else {
                    rng.gen::<H>() * settings.sharpness + settings.step - settings.sharpness
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

pub fn node_for_coordinate<T, D, Ix>(
    graph: &Graph<RegionNode<T>, RegionEdge, D, Ix>,
    point: Point2<f32>,
) -> Option<NodeIndex<Ix>>
where
    T: Default,
    D: EdgeType,
    Ix: IndexType,
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
    use crate::dual_graph::gen_dual_graph;
    use imageproc::drawing::Point as ImgPoint;
    use nalgebra::Vector2;
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
        let (mut region_graph, mut border_graph) = gen_dual_graph::<TestInner, ()>(dims, 8000, 2);

        let mut rng =
            rand_xorshift::XorShiftRng::from_seed([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);

        // Start at the center
        let center = Point2::from(dims / 2.0);
        let starting_points = (0..10)
            .map(|i| {
                let x = rng.gen_range(-500.0, 500.0);
                let y = rng.gen_range(-500.0, 500.0);
                node_for_coordinate(&region_graph, Point2::new(center.x + x, center.y + y))
                    .expect("wut")
            })
            .collect();
        let settings = default_settings_f32::<u32>(starting_points);
        visit(&mut region_graph, &mut border_graph, &settings, &mut rng);

        draw_island(
            &mut imgbuf,
            &region_graph,
            &border_graph,
            |region, border_graph| {
                //let color = region.value.height as i32;
                let elevation = region.value.elevation();
                let color = if elevation > 0.5 {
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
        RG: petgraph::visit::IntoNodeReferences,
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
