use crate::{
    dual_graph::{RegionEdge, RegionNode},
    HasElevation, HasWind,
};
use nalgebra::RealField;
use petgraph::{EdgeType, Graph};

pub struct Settings<T> {
    duration: usize, // 1 unit = 1 day
    base_speed: T,
}

pub fn visit<T, V, R, E>(
    region_graph: &mut Graph<RegionNode<V>, RegionEdge, E>,
    settings: &Settings<T>,
    rng: &mut R,
) -> Result<(), failure::Error>
where
    T: RealField,
    V: Default + HasElevation<T> + HasWind<T>,
    R: rand::Rng + ?Sized,
    E: EdgeType,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    Ok(())
}
