use crate::{
    dual_graph::{RegionEdge, RegionNode},
    HasElevation, HasWind,
};
use nalgebra::{RealField, Vector2};
use petgraph::{EdgeType, Graph};

pub struct Settings<T: RealField> {
    duration: usize, // 1 unit = 1 day
    start_speed: Option<T>,
    start_direction: Option<Vector2<T>>,
}

pub fn visit<T, V, R, E>(
    _region_graph: &mut Graph<RegionNode<V>, RegionEdge, E>,
    _settings: &Settings<T>,
    _rng: &mut R,
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

#[cfg(test)]
pub mod tests {
    #[test]
    fn wind_test() {}
}
