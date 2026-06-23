use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time;
use numpy::PyReadonlyArray1;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::algo::dijkstra;

use crate::uf::UnionFind;

#[pyclass]
pub struct UFD {
    link_list: HashMap<usize, Vec<usize>>,
    weights: HashMap<(usize, usize), f64>,
    boundary_node_left: usize,
    boundary_node_right: usize,
    uf_growth_rate: f64,
    additional_growth_rate: f64,
}

#[pymethods]
impl UFD {
    #[new]
    fn new(
        link_list: HashMap<usize, Vec<usize>>,
        weights: HashMap<(usize, usize), f64>,
        boundary_node_left: usize,
        boundary_node_right: usize,
        uf_growth_rate: f64,
        additional_growth_rate: f64,
        ) -> Self {
        UFD {
            link_list,
            weights,
            boundary_node_left,
            boundary_node_right,
            uf_growth_rate,
            additional_growth_rate,
        }
    }

    fn decode(&self,
              detection_events: PyReadonlyArray1<bool>,
              use_preskill: bool,
              additional_max_growth: f64,
              debug: bool,
              ) -> (bool, // observable
                    Option<(f64, usize)>, // preskill softoutput and #nodes with any non-zero weights
                    Option<(f64, f64, usize)>, // additional growth softoutput (simple), additional growth softoutput (cluster graph), #nodes of cluster graph
                    (f64, f64, f64, f64), // times. if not computed in each term, the term becomes zero. the last term is for debug
                    Option<Vec<(usize, usize)>>, // if debug, return fully grown edges
                    Option<HashMap<(usize, usize), f64>>, // if debug, return growths
                    ) {
        let detection_events = detection_events.as_array();
        let mut uf_de = UnionFind::new(detection_events.len() + 2);
        let mut growth = HashMap::new();
        for &key in self.weights.keys() {
            growth.insert(key, 0.);
        }

        //let mut active_dets = HashSet::new();
        //let mut de_to_active_dets = HashMap::new();
        //let mut de_to_dets = HashMap::new();
        //let mut active_de = HashSet::new();
        let mut det_to_de = HashMap::new();
        for i in 0..detection_events.len() {
            if detection_events[i] {
                //active_dets.insert(i);
                //de_to_active_dets.insert(i, vec![i; 1]);
                //active_de.insert(i);
                det_to_de.insert(i, i); // de -> de
            }
        }

        let mut all_removed_dets = HashSet::new(); // i.e., detectors completely inside some clusters

        let now = time::Instant::now();
        let mut debug_growth = 0.;
        loop {
            //println!("cnt: {}", cnt);
            debug_growth += self.uf_growth_rate;
            //if cnt > 10 {
            //    break;
            //}
            let mut complete = true;
            //for &node in &active_dets {
            for &node in det_to_de.keys() {
            //for node in 0..detection_events.len() {
                //if !det_to_de.contains_key(&node) {
                //    continue;
                //}
                let de = *det_to_de.get(&node).unwrap();
                if uf_de.size(de) % 2 == 0 || uf_de.issame(de, self.boundary_node_left) || uf_de.issame(de, self.boundary_node_right) {
                    continue;
                }
                complete = false;

                for &adj_node in self.link_list.get(&node).unwrap().iter() {
                    let edge = if node < adj_node { (node, adj_node) } else { (adj_node, node) };
                    growth.insert(edge, growth.get(&edge).unwrap() + self.uf_growth_rate);
                }
            }
            if complete { // if any growth aren't updated, the algorithm ends.
                break;
            }

            let mut added_det_to_de = HashMap::with_capacity(det_to_de.len());
            let mut removed_dets = HashSet::new();
            //for &node in &active_dets {
            for (&node, &de) in &det_to_de {
            //for node in 0..detection_events.len() {
                //if !det_to_de.contains_key(&node) {
                //    continue;
                //}

                let mut removed = true;
                for &adj_node in self.link_list.get(&node).unwrap().iter() {
                    let edge = if node < adj_node { (node, adj_node) } else { (adj_node, node) };
                    let growth_of_edge = growth.get(&edge).unwrap();
                    let weight_of_edge = self.weights.get(&edge).unwrap();
                    let diff_of_edge = weight_of_edge - growth_of_edge;
                    if diff_of_edge > 0. {
                        removed = false;
                        continue;
                    }

                    if adj_node == self.boundary_node_left || adj_node == self.boundary_node_right || detection_events[adj_node] {
                        uf_de.unite(de, adj_node);
                        //println!("{} and {} are united", de, adj_node);
                    }
                    else if det_to_de.contains_key(&adj_node) {
                        uf_de.unite(de, *det_to_de.get(&adj_node).unwrap());
                        //println!("{} and {} are united", de, *det_to_de.get(&adj_node).unwrap());
                    }
                    else if added_det_to_de.contains_key(&adj_node) {
                        uf_de.unite(de, *added_det_to_de.get(&adj_node).unwrap());
                    }
                    else if removed_dets.contains(&adj_node) || all_removed_dets.contains(&adj_node) {
                    }
                    else {
                        added_det_to_de.insert(adj_node, de);
                        //println!("{} and {} are mapped", adj_node, de);
//                        for &adj_adj_node in self.link_list.get(&adj_node).unwrap().iter() { // overgrowth is propagated to adjacent edges. `diff_of_edge` < min(weights) is assumed
//                            let edge = if adj_node < adj_adj_node { (adj_node, adj_adj_node) } else { (adj_adj_node, adj_node) };
//                            growth.insert(edge, growth.get(&edge).unwrap() + -diff_of_edge);
//                        }
                    }
                }

                if removed {
                    removed_dets.insert(node);
                }
            }
            det_to_de.extend(added_det_to_de.into_iter());
            for &removed_det in &removed_dets { // removing det leads to fast computation.
                det_to_de.remove(&removed_det);
                all_removed_dets.insert(removed_det);
            }
        }
        if debug {
            println!("debug_growth: {}", debug_growth);
            //println!("all_removed_dets: {:?}", all_removed_dets);
            println!("det_to_de.len(): {}, all_removed_dets.len(): {}", det_to_de.len(), all_removed_dets.len());
        }
        let uf_time = now.elapsed().as_secs_f64();

        let mut preskill_result = None;
        let mut preskill_time = 0.;
        if use_preskill {
            let now = time::Instant::now();
            //let mut graph = UnGraph::<usize, f64>::from_edges(&[(10, 20), (20, 30)]);
            //graph.update_edge(10.into(), 20.into(), 100.);
            //graph.update_edge(20.into(), 30.into(), 500.);
            //let node_map = dijkstra(&graph, 10.into(), Some(30.into()), |e| *e.weight());
            //println!("{:?}", node_map);
            let mut edges = Vec::with_capacity(self.weights.len());
            for &edge in self.weights.keys() {
                edges.push((edge.0 as u32, edge.1 as u32));
            }
            //let graph = UnGraph::<u32, f64>::from_edges(&self.weights.keys().cloned().collect::<Vec<(u32, u32)>>());
            //let mut graph = UnGraph::<usize, f64>::from_edges(&vec![(10,20),(20,30)]);
            let mut graph = UnGraph::<u32, f64>::from_edges(&edges);
            //let updated_weights: HashMap<(usize, usize), f64> = HashMap::new();
            for (&edge, &weight) in &self.weights {
                //updated_weights.insert(edge, (weight - growth.get(&edge).unwrap()).max(0.));
                graph.update_edge((edge.0 as u32).into(), (edge.1 as u32).into(), (weight - growth.get(&edge).unwrap()).max(0.));
            }
            //let node_map = dijkstra(&graph, NodeIndex::new(self.boundary_node_left), Some(NodeIndex::new(self.boundary_node_right)), |e| updated_weights.get(&(e.source(), e.target())));
            let node_map = dijkstra(&graph, (self.boundary_node_left as u32).into(), Some((self.boundary_node_right as u32).into()), |e| *e.weight());
            //println!("{:?}", node_map);
            let result = node_map.get(&NodeIndex::new(self.boundary_node_right)).unwrap();
            //let result = node_map.get(&self.boundary_node_right.into()).unwrap();
            //println!("preskill result: {}", result);
            let mut num_nodes_with_not_zero_weight = 2; // 2 means two boundary nodes
            for node in 0..detection_events.len() {
                for &adj_node in self.link_list.get(&node).unwrap().iter() {
                    let edge = if node < adj_node { (node, adj_node) } else { (adj_node, node) };
                    if growth.get(&edge).unwrap() < self.weights.get(&edge).unwrap() {
                        num_nodes_with_not_zero_weight += 1;
                        break;
                    }
                }
            }
            preskill_result = Some((*result, num_nodes_with_not_zero_weight));
            preskill_time = now.elapsed().as_secs_f64();
        }

        let mut additional_growth_result = None;
        let mut additional_growth_time = 0.;
        let mut debug_time = 0.;
        if additional_max_growth > 0. {
            let now = time::Instant::now();
            let mut additional_growth_when_connecting_two_boundary_nodes = None;
            let mut additional_growth = 0.;
            let mut cluster_graph = HashMap::new();
            let mut uf_de_additional = UnionFind::new(detection_events.len() + 2);
            uf_de_additional.par = uf_de.par.clone();
            uf_de_additional.siz = uf_de.siz.clone();

            let mut has_true_in_detection_events = false;
            for &det in &detection_events {
                if det {
                    has_true_in_detection_events = true;
                    break;
                }
            }
            if !has_true_in_detection_events {
                additional_growth_result = None;
            }
            else if uf_de.issame(self.boundary_node_left, self.boundary_node_right) {
                additional_growth_result = Some((0., 0., 0));
            }
            else {
                if self.boundary_node_left != uf_de.root(self.boundary_node_left) {
                    cluster_graph.insert((uf_de.root(self.boundary_node_left), self.boundary_node_left), 0.); // uf_de.root(self.boundary_node) < self.boundary_node
                }
                if self.boundary_node_right != uf_de.root(self.boundary_node_right) {
                    cluster_graph.insert((uf_de.root(self.boundary_node_right), self.boundary_node_right), 0.);
                }

                loop {
                    if additional_growth_when_connecting_two_boundary_nodes.is_none() && uf_de_additional.issame(self.boundary_node_left, self.boundary_node_right) {
                        additional_growth_when_connecting_two_boundary_nodes = Some(additional_growth);
                    }
                    if additional_growth >= additional_max_growth { // end condition
                        break;
                    }
                    additional_growth += self.additional_growth_rate;

                    let now_debug = time::Instant::now();
                    for &node in det_to_de.keys() {
                    //for node in 0..detection_events.len() {
                        //if !det_to_de.contains_key(&node) {
                        //    continue;
                        //}
                        for &adj_node in self.link_list.get(&node).unwrap().iter() {
                            let edge = if node < adj_node { (node, adj_node) } else { (adj_node, node) };
                            growth.insert(edge, growth.get(&edge).unwrap() + self.additional_growth_rate);
                        }
                    }
                    debug_time += now_debug.elapsed().as_secs_f64();

                    let mut added_det_to_de = HashMap::with_capacity(det_to_de.len());
                    let mut removed_dets = HashSet::new();
                    for &node in det_to_de.keys() {
                    //for node in 0..detection_events.len() {
                        //if !det_to_de.contains_key(&node) {
                        //    continue;
                        //}
                        let de_root = uf_de.root(*det_to_de.get(&node).unwrap()); // root can be distinguished as a node of the cluster graph because uf_de is no longer updated in the additional growth

                        let mut removed = true;
                        for &adj_node in self.link_list.get(&node).unwrap().iter() {
                            let edge = if node < adj_node { (node, adj_node) } else { (adj_node, node) };
                            let growth_of_edge = growth.get(&edge).unwrap();
                            let weight_of_edge = self.weights.get(&edge).unwrap();
                            let diff_of_edge = weight_of_edge - growth_of_edge;
                            if diff_of_edge > 0. {
                                removed = false;
                                continue;
                            }

                            if adj_node == self.boundary_node_left || adj_node == self.boundary_node_right {
                                uf_de_additional.unite(de_root, adj_node);
                                if !uf_de.issame(de_root, adj_node) {
                                    let (de_root, adj_node) = if de_root < adj_node { (de_root, adj_node) } else { (adj_node, de_root) }; // this swap could happen only if de_root is boundary_node_right and adj_node is boundary_node_left
                                    if !cluster_graph.contains_key(&(de_root, adj_node)) {
//                                        cluster_graph.insert((de_root, adj_node), additional_growth + diff_of_edge);
                                        cluster_graph.insert((de_root, adj_node), additional_growth);
                                    }
                                }
                            }
                            //else if detection_events[adj_node] {
                            //    let adj_de = uf_de.root(*det_to_de.get(&adj_node).unwrap());
                            //    if !uf_de.issame(de, adj_node) {
                            //        cluster_graph.insert((de, adj_node), additional_growth * 2);
                            //    }
                            //    //println!("{} and {} are united", det, adj_node);
                            //}
                            else if det_to_de.contains_key(&adj_node) || added_det_to_de.contains_key(&adj_node) { // adj_node is related to some detection events
                                let adj_de_root = uf_de.root(if det_to_de.contains_key(&adj_node) {
                                    *det_to_de.get(&adj_node).unwrap()
                                } else {
                                    *added_det_to_de.get(&adj_node).unwrap()
                                });
                                uf_de_additional.unite(de_root, adj_de_root);
                                if de_root != adj_de_root {
                                    let (de_root, adj_de_root) = if de_root < adj_de_root { (de_root, adj_de_root) } else { (adj_de_root, de_root) };
                                    if !cluster_graph.contains_key(&(de_root, adj_de_root)) {
//                                        cluster_graph.insert((de_root, adj_de_root), additional_growth * 2. + diff_of_edge);
                                        cluster_graph.insert((de_root, adj_de_root), additional_growth * 2.);
                                    }
                                    //println!("{} and {} are united", det, *det_to_de.get(&adj_node).unwrap());
                                }
                            }
                            else if removed_dets.contains(&adj_node) || all_removed_dets.contains(&adj_node) {
                            }
                            else {
                                added_det_to_de.insert(adj_node, de_root);
//                                for &adj_adj_node in self.link_list.get(&adj_node).unwrap().iter() { // overgrowth is propagated to adjacent edges. `diff_of_edge` < min(weights) is assumed
//                                    let edge = if adj_node < adj_adj_node { (adj_node, adj_adj_node) } else { (adj_adj_node, adj_node) };
//                                    growth.insert(edge, growth.get(&edge).unwrap() + -diff_of_edge);
//                                }
                                //println!("{} and {} are mapped", adj_node, det);
                            }
                        }
                        if removed {
                            removed_dets.insert(node);
                        }
                    }
                    det_to_de.extend(added_det_to_de.into_iter());
                    for &removed_det in &removed_dets {
                        det_to_de.remove(&removed_det);
                        all_removed_dets.insert(removed_det);
                    }
                }

                if !uf_de_additional.issame(self.boundary_node_left, self.boundary_node_right) {
                    additional_growth_result = None;
                }
                else {
                    let mut nodes = HashSet::with_capacity(detection_events.len());
                    let mut edges = Vec::with_capacity(cluster_graph.len());
                    for &key in cluster_graph.keys() {
                        edges.push((key.0 as u32, key.1 as u32));
                        nodes.insert(key.0);
                        nodes.insert(key.1);
                    }
                    //println!("cluster_graph edges: {:?}", edges);
                    let mut graph = UnGraph::<u32, f64>::from_edges(&edges);
                    for (&edge, &weight) in &cluster_graph {
                        graph.update_edge((edge.0 as u32).into(), (edge.1 as u32).into(), weight);
                    }
                    let node_map = dijkstra(&graph, (self.boundary_node_left as u32).into(), Some((self.boundary_node_right as u32).into()), |e| *e.weight());
                    //println!("{:?}", node_map);
                    let result = node_map.get(&NodeIndex::new(self.boundary_node_right)).unwrap();
                    additional_growth_result = Some((additional_growth_when_connecting_two_boundary_nodes.unwrap(), *result, nodes.len()))//graph.node_count())); // graph.node_count() seems broken, and it returns the maximum node index + 1.
                }
            }
            additional_growth_time = now.elapsed().as_secs_f64();
        }
        if debug {
            //println!("all_removed_dets after additional growth: {:?}", all_removed_dets);
            println!("det_to_de.len(): {}, all_removed_dets.len(): {} after additional growth", det_to_de.len(), all_removed_dets.len());
        }

        //println!("88: {}", uf_de.size(88));
        //println!("88: {}", uf_de.size(88));
        //println!("111: {}", uf_de.size(111));
        //println!("114: {}", uf_de.size(114));
        //println!("self.boundary_node_left({}): {}", self.boundary_node_left, uf_de.size(self.boundary_node_left));
        //println!("self.boundary_node_right({}): {}", self.boundary_node_right, uf_de.size(self.boundary_node_right));
        //return ((uf_de.size(self.boundary_node_left)-1) % 2 == 1, growth) // uf_de.size(self.boundary_node_left) contains self.boundary_node_left itself, so -1

        ((uf_de.size(self.boundary_node_left)-1) % 2 == 1,
            preskill_result,
            additional_growth_result,
            (uf_time, preskill_time, additional_growth_time, debug_time),
            if debug { Some(growth.iter().filter(|(&key, &value)| value >= *self.weights.get(&key).unwrap()).map(|(&key, &value)| key).collect()) } else { None },
            if debug { Some(growth) } else { None },
            )
    }
}
