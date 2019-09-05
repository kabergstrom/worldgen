//! Peak Island Height
//! Visits the graph, traversing every node and assigning a new elevation. This algorithm takes a starting location
//! and elevation, and progressively lowers neighbors outwards from the starting location until a threshold is reached.
//!
//!
use crate::{
    dual_graph::{RegionEdge, RegionNode, RegionNodeIdx},
    HasElevation,
};
use petgraph::{
    graph::{IndexType, NodeIndex},
    EdgeType, Graph,
};
use std::collections::HashSet;

use num::Zero;
use std::ops::{Add, Mul, Sub};

#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Settings<T, Ix: IndexType> {
    starting_node: NodeIndex<Ix>,
    starting_elevation: T,
    end_elevation: T,
    radius: T,
    sharpness: T,
    step: T,
}

fn default_settings_f32<Ix: IndexType>(starting_node: NodeIndex<Ix>) -> Settings<f32, Ix> {
    Settings {
        starting_node,
        starting_elevation: 1.0,
        radius: 0.95,
        end_elevation: 0.0001,
        sharpness: 0.2,
        step: 0.1,
    }
}
fn default_settings_f64<Ix: IndexType>(starting_node: NodeIndex<Ix>) -> Settings<f64, Ix> {
    Settings {
        starting_node,
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

pub fn visit<T, R, H, D, Ix>(
    graph: &mut Graph<RegionNode<T>, RegionEdge, D, Ix>,
    settings: Settings<H, Ix>,
    rng: &mut R,
) -> Result<(), failure::Error>
where
    H: Ord + PartialEq + Add<Output = H> + Mul<Output = H> + Sub<Output = H> + Zero + Copy + Sized,
    T: Default + HasElevation<H>,
    R: rand::Rng + ?Sized,
    D: EdgeType,
    Ix: IndexType,
    rand::distributions::Standard: rand::distributions::Distribution<H>,
{
    let mut completed = HashSet::with_capacity(graph.node_count());
    let mut queue = Vec::with_capacity(graph.node_count());

    let mut current_elevation = settings.starting_elevation;
    fetch_or_err_mut(graph, settings.starting_node)?
        .value
        .set_elevation(current_elevation);

    let mut i = 0;
    while i < queue.len() && current_elevation >= settings.end_elevation {
        let current_node = queue[i];
        let parent_elevation = fetch_or_err(graph, queue[i])?.value.elevation();
        current_elevation = parent_elevation * settings.radius;

        let mut walker = graph.neighbors(current_node).detach();
        while let Some(neighbor_idx) = walker.next(&graph) {
            if !completed.contains(&neighbor_idx.1) {
                let modifier = if settings.sharpness == num::zero() {
                    parent_elevation
                } else {
                    rng.gen::<H>() * settings.sharpness + settings.step - settings.sharpness
                };

                fetch_or_err_mut(graph, settings.starting_node)?
                    .value
                    .set_elevation(current_elevation * modifier);

                queue.push(neighbor_idx.1);
                completed.insert(neighbor_idx.1);
            }
        }
        i += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual_graph::{gen_dual_graph, tests::draw_graph};
    use nalgebra::Vector2;

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
        let (region_graph, border_graph) = gen_dual_graph::<TestInner, ()>(dims, 6500, 2);

        imgbuf.save("graphs.png").unwrap();
    }
}
