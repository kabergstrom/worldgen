#![deny(clippy::pedantic)]
#![allow(dead_code, clippy::module_name_repetitions)]
use nalgebra::{RealField, Vector2};

pub mod dual_graph;
pub mod peak_automata;
pub mod simple_wind;

pub trait HasValue {
    type Value;

    fn value(&self) -> &Self::Value;
    fn value_mut(&mut self) -> &mut Self::Value;
}

pub trait HasElevation<T: RealField> {
    fn elevation(&self) -> T;
    fn set_elevation(&mut self, height: T);
}

pub trait HasMoisture<T: RealField> {
    fn moisture(&self) -> T;
    fn set_moisture(&mut self, moisture: T);
}

pub trait HasTemperature<T: RealField> {
    fn temperature(&self) -> T;
    fn set_temperature(&mut self, temperature: T);
}

pub trait HasWind<T: RealField> {
    fn vector(&self) -> Vector2<T>;
    fn set_vector(&mut self, vector: Vector2<T>);
}

pub trait AnnualRainfall<T: RealField> {
    fn annual_rainfaill(&self) -> T {
        unimplemented!()
    }
}
