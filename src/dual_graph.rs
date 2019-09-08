use nalgebra::{Point2, Vector2};
use petgraph::graph::NodeIndex;
use rand::Rng;
use std::collections::HashMap;
use voronoi::voronoi;

pub type BorderNodeIdx = NodeIndex;
pub type BorderEdgeIdx = petgraph::graph::EdgeIndex;
pub type RegionNodeIdx = NodeIndex;
pub type RegionEdgeIdx = petgraph::graph::EdgeIndex;

#[derive(Debug)]
pub struct BorderNode<T = ()> {
    pub regions: Vec<RegionNodeIdx>,
    pub pos: Point2<f32>,
    pub value: T,
}
#[derive(Debug)]
pub struct BorderEdge {
    pub region_edge: Option<RegionEdgeIdx>,
    pub regions: Vec<RegionNodeIdx>,
}
#[derive(Debug)]
pub struct RegionNode<T = ()> {
    pub borders: Vec<BorderNodeIdx>,
    pub pos: Point2<f32>,
    pub value: T,
}
#[derive(Debug)]
pub struct RegionEdge {
    pub border_edge: Option<BorderEdgeIdx>,
    pub borders: Vec<BorderNodeIdx>,
}
pub type RegionGraph<T = ()> = petgraph::graph::UnGraph<RegionNode<T>, RegionEdge>;
pub type BorderGraph<T = ()> = petgraph::graph::UnGraph<BorderNode<T>, BorderEdge>;

fn poly_centroids(diagram: &voronoi::DCEL) -> Vec<voronoi::Point> {
    let mut face_centroids = vec![voronoi::Point::new(0.0, 0.0); diagram.faces.len()];
    let mut num_face_vertices = vec![0; diagram.faces.len()];
    for edge in diagram.halfedges.iter() {
        if !edge.alive {
            continue;
        }
        let pt = diagram.vertices[edge.origin].coordinates;
        let face_pt = face_centroids[edge.face];
        face_centroids[edge.face] = voronoi::Point::new(
            face_pt.x.into_inner() + pt.x.into_inner(),
            face_pt.y.into_inner() + pt.y.into_inner(),
        );
        num_face_vertices[edge.face] += 1;
    }
    for i in 0..num_face_vertices.len() {
        let num_vertices = num_face_vertices[i];
        let face_pt = face_centroids[i];
        face_centroids[i] = voronoi::Point::new(
            face_pt.x.into_inner() / f64::from(num_vertices),
            face_pt.y.into_inner() / f64::from(num_vertices),
        );
    }

    face_centroids.remove(face_centroids.len() - 1);
    face_centroids
}

fn gen_points(count: usize, bounds: &voronoi::Point) -> Vec<voronoi::Point> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let x: f64 = rng.sample(rand::distributions::Standard);
            let y: f64 = rng.sample(rand::distributions::Standard);
            voronoi::Point::new(x * bounds.x.into_inner(), y * bounds.y.into_inner())
        })
        .collect()
}

fn gen_voronoi(
    dims: voronoi::Point,
    num_points: usize,
    num_lloyd_iterations: u32,
) -> voronoi::DCEL {
    let points = gen_points(num_points, &dims);
    let mut vor_diagram;
    let mut points: Vec<voronoi::Point> = points;
    let mut i = 0;
    loop {
        vor_diagram = voronoi(points.clone(), dims.x.into());
        if i == num_lloyd_iterations {
            break;
        }
        points = poly_centroids(&vor_diagram);
        i += 1;
    }
    vor_diagram
}

fn get_or_insert_border_node<T>(
    border_node_map: &mut HashMap<usize, BorderNodeIdx>,
    graph: &mut BorderGraph<T>,
    diagram: &voronoi::DCEL,
    idx: usize,
) -> BorderNodeIdx
where
    T: Default,
{
    if let Some(border_node) = border_node_map.get(&idx) {
        *border_node
    } else {
        let pos = diagram.vertices[idx].coordinates;
        let border_node = graph.add_node(BorderNode {
            regions: Vec::new(),
            pos: Point2::new(pos.x.into_inner() as f32, pos.y.into_inner() as f32),
            value: Default::default(),
        });
        border_node_map.insert(idx, border_node);
        border_node
    }
}
fn get_or_insert_region_node<T>(
    region_node_map: &mut HashMap<usize, RegionNodeIdx>,
    graph: &mut RegionGraph<T>,
    pos: Point2<f32>,
    idx: usize,
) -> RegionNodeIdx
where
    T: Default,
{
    if let Some(region_node) = region_node_map.get(&idx) {
        *region_node
    } else {
        let region_node = graph.add_node(RegionNode {
            borders: Vec::new(),
            pos,
            value: Default::default(),
        });
        region_node_map.insert(idx, region_node);
        region_node
    }
}

pub fn gen_dual_graph<R, B>(
    dims: Vector2<f32>,
    num_points: usize,
    num_lloyd_iterations: u32,
) -> (RegionGraph<R>, BorderGraph<B>)
where
    R: Default,
    B: Default,
{
    let vor_diagram = gen_voronoi(
        voronoi::Point::new(f64::from(dims.x), f64::from(dims.y)),
        num_points,
        num_lloyd_iterations,
    );

    let mut region_graph = RegionGraph::<R>::new_undirected();
    let mut border_graph = BorderGraph::<B>::new_undirected();
    let mut border_node_map: HashMap<usize, BorderNodeIdx> = HashMap::new();
    let mut region_node_map: HashMap<usize, RegionNodeIdx> = HashMap::new();
    for (i, face) in vor_diagram
        .faces
        .iter()
        .take(vor_diagram.faces.len() - 1)
        .enumerate()
    {
        let region_node_idx = get_or_insert_region_node(
            &mut region_node_map,
            &mut region_graph,
            Point2::new(0.0, 0.0),
            i,
        );
        let region_node = &mut region_graph[region_node_idx];
        let mut curr_edge = face.outer_component;
        let mut prev_edge;
        let mut pos = Point2::new(0.0, 0.0);
        let mut num_edges = 0;
        loop {
            prev_edge = curr_edge;
            curr_edge = vor_diagram.halfedges[curr_edge].next;
            let border_idx = get_or_insert_border_node(
                &mut border_node_map,
                &mut border_graph,
                &vor_diagram,
                vor_diagram.halfedges[curr_edge].origin,
            );
            region_node.borders.push(border_idx);
            let next_idx = get_or_insert_border_node(
                &mut border_node_map,
                &mut border_graph,
                &vor_diagram,
                vor_diagram.halfedges[prev_edge].origin,
            );
            let edge_idx =
                if let Some((e, _)) = border_graph.find_edge_undirected(border_idx, next_idx) {
                    e
                } else {
                    border_graph.add_edge(
                        border_idx,
                        next_idx,
                        BorderEdge {
                            region_edge: None,
                            regions: Vec::new(),
                        },
                    )
                };
            border_graph[edge_idx].regions.push(region_node_idx);
            num_edges += 1;
            let vertex_pos =
                vor_diagram.vertices[vor_diagram.halfedges[curr_edge].origin].coordinates;
            pos = Point2::new(
                pos.x + vertex_pos.x.into_inner() as f32,
                pos.y + vertex_pos.y.into_inner() as f32,
            );
            if curr_edge == face.outer_component {
                break;
            }
        }
        region_node.pos = pos / num_edges as f32;
    }
    use petgraph::visit::EdgeRef;
    for edge in border_graph.edge_references() {
        let regions = &edge.weight().regions;
        if regions.len() > 1 {
            assert!(regions.len() == 2);
            let region_a = regions[0];
            let region_b = regions[1];
            if region_graph
                .find_edge_undirected(region_a, region_b)
                .is_none()
            {
                region_graph.add_edge(
                    region_a,
                    region_b,
                    RegionEdge {
                        border_edge: Some(edge.id()),
                        borders: vec![edge.source(), edge.target()],
                    },
                );
            }
        }
    }
    for edge in region_graph.edge_references() {
        let borders = &edge.weight().borders;
        assert!(borders.len() == 2);
        let border_a = borders[0];
        let border_b = borders[1];
        let (edge_idx, _) = border_graph
            .find_edge_undirected(border_a, border_b)
            .expect("border edge did not exist");
        let border_edge = &mut border_graph[edge_idx];
        border_edge.region_edge.replace(edge.id());
    }
    (region_graph, border_graph)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[test]
    pub fn gen_dual_graph_test() {
        let dims = Vector2::new(1024.0, 1024.0);
        let mut imgbuf = image::ImageBuffer::from_pixel(
            dims.x as u32,
            dims.y as u32,
            image::Rgb([222, 222, 222]),
        );
        let (region_graph, border_graph) = gen_dual_graph::<(), ()>(dims, 6500, 2);
        draw_graph(
            &mut imgbuf,
            &region_graph,
            |n| (image::Rgb([0, 0, 222]), n.pos, 2),
            |e| {
                use petgraph::visit::EdgeRef;
                let source_node = &region_graph[e.source()];
                let target_node = &region_graph[e.target()];
                (image::Rgb([0, 0, 222]), source_node.pos, target_node.pos)
            },
        );
        draw_graph(
            &mut imgbuf,
            &border_graph,
            |n| (image::Rgb([222, 0, 0]), n.pos, 2),
            |e| {
                use petgraph::visit::EdgeRef;
                let source_node = &border_graph[e.source()];
                let target_node = &border_graph[e.target()];
                (image::Rgb([0, 222, 0]), source_node.pos, target_node.pos)
            },
        );
        for edge in border_graph.edge_references() {
            use petgraph::visit::EdgeRef;
            let regions = &edge.weight().regions;
            let pos_a = border_graph[edge.source()].pos;
            let pos_b = border_graph[edge.target()].pos;
            let pos = Point2::from((pos_a.coords + pos_b.coords) / 2.0);

            for region in regions.iter() {
                let pos_region = region_graph[*region].pos;
                imageproc::drawing::draw_antialiased_line_segment_mut(
                    &mut imgbuf,
                    (pos.x as i32, pos.y as i32),
                    (pos_region.x as i32, pos_region.y as i32),
                    image::Rgb([0, 222, 222]),
                    imageproc::pixelops::interpolate,
                );
            }
        }
        for edge in region_graph.edge_references() {
            use petgraph::visit::EdgeRef;
            let borders = &edge.weight().borders;
            let pos_a = region_graph[edge.source()].pos;
            let pos_b = region_graph[edge.target()].pos;
            let pos = Point2::from((pos_a.coords + pos_b.coords) / 2.0);

            for border in borders.iter() {
                let pos_border = border_graph[*border].pos;
                imageproc::drawing::draw_antialiased_line_segment_mut(
                    &mut imgbuf,
                    (pos.x as i32, pos.y as i32),
                    (pos_border.x as i32, pos_border.y as i32),
                    image::Rgb([0, 222, 222]),
                    imageproc::pixelops::interpolate,
                );
            }
        }
        imgbuf.save("output/graphs.png").unwrap();
    }

    pub(crate) fn draw_graph<
        G: petgraph::visit::IntoNodeReferences + petgraph::visit::IntoEdgeReferences,
        N: Fn(
            &<G as petgraph::visit::Data>::NodeWeight,
        ) -> (<I as image::GenericImageView>::Pixel, Point2<f32>, i32),
        E: Fn(
            <G as petgraph::visit::IntoEdgeReferences>::EdgeRef,
        ) -> (
            <I as image::GenericImageView>::Pixel,
            Point2<f32>,
            Point2<f32>,
        ),
        I,
    >(
        imgbuf: &mut I,
        graph: G,
        node_color: N,
        edge_color: E,
    ) where
        I: image::GenericImage,
        I::Pixel: 'static,
        <<I as image::GenericImageView>::Pixel as image::Pixel>::Subpixel:
            conv::ValueInto<f32> + imageproc::definitions::Clamp<f32>,
    {
        use petgraph::visit::NodeRef;
        for node in graph.node_references() {
            let (color, pt, size) = node_color(node.weight());
            imageproc::drawing::draw_filled_circle_mut(
                imgbuf,
                (pt.x as i32, pt.y as i32),
                size,
                color,
            );
        }
        for edge in graph.edge_references() {
            let (color, from, to) = edge_color(edge);
            imageproc::drawing::draw_antialiased_line_segment_mut(
                imgbuf,
                (from.x as i32, from.y as i32),
                (to.x as i32, to.y as i32),
                color,
                imageproc::pixelops::interpolate,
            );
        }
    }
}
