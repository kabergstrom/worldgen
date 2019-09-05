pub mod dual_graph;
pub mod peak_island_height;

pub trait HasElevation<T> {
    fn elevation(&self) -> T;
    fn set_elevation(&mut self, height: T);
}

pub trait HasMoisture<T>
where
    T: Ord + PartialEq,
{
    fn moisture() -> T;
}
