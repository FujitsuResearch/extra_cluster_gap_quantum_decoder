use pyo3::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time;
use std::cmp::Reverse;
use std::hash::BuildHasherDefault;
use numpy::PyReadonlyArray1;
use petgraph::graph::{NodeIndex, UnGraph};
//use petgraph::algo::dijkstra;
use radix_heap::RadixHeapMap;
use rustc_hash::FxHasher;

use crate::uf::UnionFind;
use crate::dijkstra::dijkstra;
use crate::bounded_dijkstra::bounded_dijkstra;

type Hasher = BuildHasherDefault<FxHasher>;

#[pyclass]
pub struct UFD2 {
    link_list: HashMap<usize, Vec<usize>>,
    weights: HashMap<(usize, usize), u64>,
    boundary_node_left: usize,
    boundary_node_right: usize,
}

#[pymethods]
impl UFD2 {
    #[new]
    fn new(
        link_list: HashMap<usize, Vec<usize>>,
        weights: HashMap<(usize, usize), f64>,
        boundary_node_left: usize,
        boundary_node_right: usize,
        convert_coef: f64,
        ) -> Self {
        let weights = weights
                .iter()
                .map(|(&key, &value)| (key, ((value * convert_coef).round() as u64) * 4))
                .collect();
        UFD2 {
            link_list,
            weights,
            boundary_node_left,
            boundary_node_right,
        }
    }

    fn decode(&self,
              detection_events: PyReadonlyArray1<bool>,
              use_preskill: bool,
              use_bounded_dijkstra: bool,
              additional_max_growth: u64,
              debug: bool,
              ) -> (bool, // observable
                    Option<(u64, usize, usize)>, // preskill softoutput, #nodes with any non-zero weights, and actually visited nodes
                    Option<(Option<u64>, usize, usize)>, // bounded dijkstra softoutput, #nodes with any non-zero weights, and actually visited nodes
                    Option<(u64, u64, usize, usize)>, // additional growth softoutput (simple), additional growth softoutput (cluster graph), #nodes of cluster graph, and actually visited nodes
                    (f64, f64, f64, f64, f64), // times. if not computed in each term, the term becomes zero. the last term is for debug
                    Option<Vec<(usize, usize)>>, // if debug, return fully grown edges
                    Option<HashMap<(usize, usize), u64>>, // if debug, return growths
                    Option<HashMap<usize, usize>>, // if debug, return det_to_de (but cloned)
                    Option<HashSet<usize>>, // if debug, return additional_det
                    Option<usize>, // if debug, return number of additional collision
                    Option<u64>, // return grown weights in union find decoder
                    ) {
        assert!(!use_bounded_dijkstra || additional_max_growth > 0, "additional_max_growth must be greater than 0 if use_bounded_dijkstra is true");

        let detection_events = detection_events.as_array();
        let mut det_to_de = HashMap::new();
        let mut ufroot_to_end_edges = HashMap::new();
        let mut active_roots = HashSet::new();
        for i in 0..detection_events.len() {
            if detection_events[i] {
                det_to_de.insert(i, i); // de -> de

                ufroot_to_end_edges.insert(i, HashSet::new());
                let end_edges = ufroot_to_end_edges.get_mut(&i).unwrap();
                for &adj_node in self.link_list.get(&i).unwrap().iter() {
                    //let edge = common_edge((i, adj_node));
                    let edge = (i, adj_node);
                    end_edges.insert(edge);
                }

                active_roots.insert(i);
            }
        }
        if det_to_de.len() == 0 {
            return (false,
                None,//preskill_result,
                None,//bounded_dijkstra_result,
                None,//additional_growth_result,
                (0., 0., 0., 0., 0.),//(uf_time, preskill_time, additional_growth_time, debug_time),
                None,
                None,
                None,
                None,
                None,
                None,
                )
        }

        let mut uf_de = UnionFind::new(detection_events.len() + 2);

        let mut event_map: HashMap<(usize, usize), u64> = HashMap::new();
        for (&node, &de) in &det_to_de {
            for &adj_node in self.link_list.get(&node).unwrap().iter() {
                let edge = (node, adj_node);
                let cedge = common_edge(edge);
                let &weight = self.weights.get(&cedge).unwrap();
                if let Some(&inverse_edge_value) = event_map.get(&inverse_edge(edge)) {
                    event_map.insert(edge, inverse_edge_value / 2);
                    event_map.insert(inverse_edge(edge), inverse_edge_value / 2);
                }
                else {
                    event_map.insert(edge, weight);
                }
            }
        }

        let mut events_cnt = HashMap::new();
        //let mut events = BinaryHeap::new();
        let mut events = RadixHeapMap::new();
        for (&edge, &t) in &event_map {
            //events.push(Reverse((t, edge)));
            events.push(Reverse(t), edge);
            *events_cnt.entry(t).or_insert(0) += 1;
        }
        //events.constrain(); // In RadixHeapMap, it is required to use `top`

        let mut growths = HashMap::new();
        let now = time::Instant::now();
        let mut debug_time = 0.;
        let mut prev_peek_global_t = None;
        loop {
            if active_roots.len() == 0 {
                if prev_peek_global_t.is_none() {
                    break;
                }
                else if *events_cnt.get(&prev_peek_global_t.unwrap()).unwrap() == 0 {
                    break;
                }
                //print!("{}", prev_peek_global_t.unwrap());
            }
            if events.len() == 0 {
                break;
            }
            //let peek_global_t = events.peek().unwrap().0.0;
            events.constrain();
            let peek_global_t = events.top().unwrap().0;
            let mut updated_roots = HashSet::new();
            loop {
                //println!("events: {:?}", events);
                //println!("event_map: {:?}", event_map);
                //println!("active_roots: {:?}", active_roots);
                //println!("growths: {:?}", growths);
                //println!("ufroot_to_end_edges: {:?}", ufroot_to_end_edges);
                //println!("updated_roots: {:?}", updated_roots);
                //if active_roots.len() == 0 {
                //    if prev_peek_global_t.is_none() {
                //        break;
                //    }
                //    else if *events_cnt.get(&prev_peek_global_t.unwrap()).unwrap() == 0 {
                //        break;
                //    }
                //}
                //if updated_roots.len() > 0 {
                //    break;
                //}
                if events.len() == 0 {
                    break;
                }
                //if events.peek().unwrap().0.0 != peek_global_t { // BinaryHeap can do this But, RadixHeapMap cannot do this! So, it do below method
                //events.constrain(); // it breaks!!!
                //if events.top().unwrap().0 != peek_global_t {
                //    break;
                //}
                if *events_cnt.get(&peek_global_t).unwrap() == 0 {
                    break;
                }

                // BinaryHeap
                //let peek_event = events.pop().unwrap().0; // .0 is due to Reverse
                //let global_t = peek_event.0;
                // RadixHeapMap
                let peek_event = events.pop().unwrap();
                let global_t = peek_event.0.0;
                assert!(global_t == peek_global_t);
                let edge = peek_event.1;
                *events_cnt.entry(global_t).or_insert(0) -= 1;
                //if global_t != peek_global_t { // top of RadixHeapMap returns the peek including one before popped value! So, instead of above method, do it
                //    events.push(Reverse(global_t), edge);
                //    break;
                //}
                if !event_map.contains_key(&edge) {
                    continue;
                }
                if *event_map.get(&edge).unwrap() != global_t { // popped value is too old to use
                    continue;
                }

                let cedge = common_edge(edge);
                let iedge = inverse_edge(edge);
                growths.insert(cedge, *self.weights.get(&cedge).unwrap());
                event_map.remove(&edge);
                event_map.remove(&iedge);
                let de = uf_de.root(*det_to_de.get(&edge.0).unwrap());
                let end_edges = ufroot_to_end_edges.get_mut(&de).unwrap();
                end_edges.remove(&edge);
                end_edges.remove(&inverse_edge(edge));

                if debug {
                    //if edge == (0, 3) {
                    //    println!("(0, 3)!!");
                    //}
                    //if edge == (6, 3) {
                    //    println!("(6, 3)!!");
                    //}
                    //if edge == (3, 0) {
                    //    println!("(3, 0)!!");
                    //}
                    //if edge == (3, 6) {
                    //    println!("(3, 6)!!");
                    //}
                }

                if edge.1 == self.boundary_node_left || edge.1 == self.boundary_node_right {
                    if debug {
                        //if edge.0 == 65 {
                        //    println!("65 yeah!, de: {}, edge.1: {}", de, edge.1);
                        //}
                        //else if edge.0 == 14 {
                        //    println!("14 yeah!, de: {}, edge.1: {}", de, edge.1);
                        //}
                        //else if edge.0 == 98 {
                        //    println!("98 yeah!, de: {}, edge.1: {}", de, edge.1);
                        //}
                    }
                    let boundary_root = uf_de.root(edge.1);
                    if debug {
                        //println!("uf_de.root({}): {}, uf_de.issame(edge.1, de): {}", edge.1, uf_de.root(edge.1), uf_de.issame(edge.1, de));
                    }
                    uf_de.unite(de, edge.1);
                    if debug {
                        //println!("uf_de.root({}): {}, uf_de.issame(edge.1, de): {}", edge.1, uf_de.root(edge.1), uf_de.issame(edge.1, de));
                    }
                    let current_root = uf_de.root(edge.1);
                    if boundary_root == current_root && de != current_root {
                        let removed_end_edges = ufroot_to_end_edges.remove(&de).unwrap();
                        let end_edges = ufroot_to_end_edges.get_mut(&current_root).unwrap();
                        end_edges.extend(removed_end_edges);
                    }
                    else if boundary_root != edge.1 && boundary_root != current_root {
                        let removed_end_edges = ufroot_to_end_edges.remove(&boundary_root).unwrap();
                        let end_edges = ufroot_to_end_edges.get_mut(&de).unwrap();
                        end_edges.extend(removed_end_edges);
                    }

                    updated_roots.insert(current_root);
                    active_roots.remove(&de);
                }
                else if let Some(&adj_de) = det_to_de.get(&edge.1) {
                    if !uf_de.issame(de, adj_de) {
                        let adj_de = uf_de.root(adj_de);
                        uf_de.unite(de, adj_de);

                        let new_root = uf_de.root(de);
                        if de == new_root {
                            let adj_end_edges = ufroot_to_end_edges.remove(&adj_de).unwrap();
                            let end_edges = ufroot_to_end_edges.get_mut(&de).unwrap();
                            end_edges.extend(adj_end_edges);
                            end_edges.remove(&iedge);
                            active_roots.remove(&adj_de);
                        }
                        else {
                            let end_edges = ufroot_to_end_edges.remove(&de).unwrap();
                            let adj_end_edges = ufroot_to_end_edges.get_mut(&adj_de).unwrap();
                            adj_end_edges.extend(end_edges.into_iter());
                            adj_end_edges.remove(&iedge);
                            active_roots.remove(&de);
                        }
                        active_roots.insert(new_root);
                        updated_roots.insert(new_root);
                    }
                }
                else {
                    det_to_de.insert(edge.1, de);
                    for &adj_node in self.link_list.get(&edge.1).unwrap().iter() {
                        let end_edge = (edge.1, adj_node);
                        if end_edge == edge || inverse_edge(end_edge) == edge {
                            continue;
                        }

                        let end_cedge = common_edge(end_edge);
                        let end_iedge = inverse_edge(end_edge);

                        let weight = self.weights.get(&end_cedge).unwrap();
                        let growth = growths.get(&end_cedge).unwrap_or(&0);
                        if growth >= weight {
                            continue;
                        }

                        end_edges.insert(end_edge);

                        if let Some(edge_value) = event_map.get(&end_edge) {
                            continue;
                        }
                        else if let Some(inverse_edge_value) = event_map.get(&end_iedge) {
                            //println!("end_edge: {:?} and end_iedge: {:?} is updated", end_edge, end_iedge);
                            let updated_growth = weight - (inverse_edge_value - peek_global_t);
                            growths.insert(end_cedge, updated_growth);
                            let val = peek_global_t + (weight - updated_growth) / 2;
                            event_map.insert(end_edge, val);
                            event_map.insert(end_iedge, val);
                            //events.push(Reverse((val, end_edge)));
                            //events.push(Reverse((val, end_iedge)));
                            assert!(val >= peek_global_t);
                            //println!("hoge");
                            //println!("val: {}, peek_global_t: {}, top: {}, global_t: {}", val, peek_global_t, events.top().unwrap().0, global_t);
                            events.push(Reverse(val), end_edge);
                            events.push(Reverse(val), end_iedge);
                            //println!("piyo");
                            *events_cnt.entry(val).or_insert(0) += 2;
                        }
                        else {
                            //println!("end_edge: {:?} is updated", end_edge);
                            let val = peek_global_t + weight - growth;
                            event_map.insert(end_edge, val);
                            //events.push(Reverse((val, end_edge)));
                            assert!(val >= peek_global_t);
                            events.push(Reverse(val), end_edge);
                            *events_cnt.entry(val).or_insert(0) += 1;
                        }
                    }
                }
            }
            //if active_roots.len() == 0 { // to be simple for preskill and additional growth, this small optimization is removed
            //    break;
            //}

            //println!("updated_roots before collect: {:?}", updated_roots);
            //let updated_roots: HashSet<_> = updated_roots.into_iter().map(|de| uf_de.root(de)).collect();
            //println!("updated_roots after collect: {:?}", updated_roots);
            //assert!(peek_global_t )

            let debug_now = time::Instant::now();
            for &de in &updated_roots {
                if uf_de.size(de) % 2 == 0 || uf_de.issame(de, self.boundary_node_left) || uf_de.issame(de, self.boundary_node_right) {
                    if let Some(end_edges) = ufroot_to_end_edges.get(&de) {
                        for &end_edge in end_edges {
                            let end_cedge = common_edge(end_edge);
                            let weight = self.weights.get(&end_cedge).unwrap();
                            let end_iedge = inverse_edge(end_edge);

                            if let Some(edge_value) = event_map.get(&end_edge) {
                                if let Some(inverse_edge_value) = event_map.get(&end_iedge) {
                                    assert_eq!(edge_value, inverse_edge_value); // if existed, always be the same

                                    //println!("2x; end_edge: {:?}, edge_value: {}, peek_global_t: {}, weight: {}", end_edge, edge_value, peek_global_t, weight);
                                    let updated_growth = weight - (edge_value - peek_global_t) * 2;
                                    growths.insert(end_cedge, updated_growth);
                                    let val = peek_global_t + weight - updated_growth;
                                    event_map.insert(end_iedge, val);
                                    //events.push(Reverse((val, end_iedge)));
                                    assert!(val >= peek_global_t);
                                    events.push(Reverse(val), end_iedge);
                                    *events_cnt.entry(val).or_insert(0) += 1;
                                }
                                else {
                                    //println!("1x; end_edge: {:?}, edge_value: {}, peek_global_t: {}, weight: {}", end_edge, edge_value, peek_global_t, weight);
                                    let val = weight - (edge_value - peek_global_t);
                                    //assert!(val >= peek_global_t);
                                    growths.insert(end_cedge, val);
                                }
                                event_map.remove(&end_edge);
                            }
                            // if only inverse edge's event_map exists, do nothing here.
                        }
                    }
                    active_roots.remove(&de);
                }
                else {
                    if let Some(end_edges) = ufroot_to_end_edges.get(&de) {
                        for &end_edge in end_edges {
                            let end_cedge = common_edge(end_edge);
                            let weight = self.weights.get(&end_cedge).unwrap();
                            let end_iedge = inverse_edge(end_edge);

                            if !event_map.contains_key(&end_edge) {
                                if let Some(inverse_edge_value) = event_map.get(&end_iedge) {
                                    let updated_growth = weight - (inverse_edge_value - peek_global_t);
                                    //println!("2x; end_edge: {:?}, inverse_edge_value: {}, peek_global_t: {}, weight: {}", end_edge, inverse_edge_value, peek_global_t, weight);
                                    growths.insert(end_cedge, updated_growth);
                                    let val = peek_global_t + (weight - updated_growth) / 2;
                                    event_map.insert(end_edge, val);
                                    event_map.insert(end_iedge, val);
                                    //events.push(Reverse((val, end_edge)));
                                    //events.push(Reverse((val, end_iedge)));
                                    assert!(val >= peek_global_t);
                                    //println!("val: {}, peek_global_t: {}, top: {}", val, peek_global_t, events.top().unwrap().0);
                                    events.push(Reverse(val), end_edge);
                                    events.push(Reverse(val), end_iedge);
                                    //println!("piyo");
                                    *events_cnt.entry(val).or_insert(0) += 2;
                                }
                                else {
                                    let growth = growths.get(&end_cedge).unwrap_or(&0);
                                    //println!("1x; end_edge: {:?}, growth: {}, peek_global_t: {}, weight: {}", end_edge, growth, peek_global_t, weight);
                                    let val = peek_global_t + weight - growth;
                                    event_map.insert(end_edge, val);
                                    //events.push(Reverse((val, end_edge)));
                                    assert!(val >= peek_global_t);
                                    events.push(Reverse(val), end_edge);
                                    *events_cnt.entry(val).or_insert(0) += 1;
                                }
                            }
                            // if edge's event_map exists, do nothing here.
                        }
                    }
                }
            }
            prev_peek_global_t = Some(peek_global_t);
            //debug_time += debug_now.elapsed().as_secs_f64();
        }
        let uf_time = now.elapsed().as_secs_f64();

        // to set growths correctly
        //let prev_peek_global_t = prev_peek_global_t.unwrap();
        //loop {
        //    let event = events.pop();
        //    if let Some(event) = event {
        //        let edge = event.1;
        //        if let Some(edge_value) = event_map.get(&edge) {
        //            let cedge = common_edge(edge);
        //            let iedge = inverse_edge(edge);
        //            let weight = self.weights.get(&cedge).unwrap();
        //            let val = weight - (edge_value - prev_peek_global_t);
        //            growths.insert(cedge, val);
        //            event_map.remove(&edge);
        //            event_map.remove(&iedge);
        //        }
        //    }
        //    else {
        //        break;
        //    }
        //}

        let mut preskill_result = None;
        let mut preskill_time = 0.;
        if use_preskill {
            if debug {
                //let peek_global_t = events.top().unwrap().0;
                //println!("peek_global_t: {}, event_cnt[peek_global_t]: {}, heap: {:?}", peek_global_t, events_cnt.get(&peek_global_t).unwrap(), events);
                //println!("growths: {:?}", growths);
                //println!("weights: {:?}", self.weights);
            }
            let mut edges = Vec::with_capacity(self.weights.len());
            for &edge in self.weights.keys() {
                edges.push((edge.0 as u32, edge.1 as u32));
            }
            let mut graph = UnGraph::<u32, u64>::from_edges(&edges);
            let now = time::Instant::now();
            for (&edge, &weight) in &self.weights {
                graph.update_edge((edge.0 as u32).into(), (edge.1 as u32).into(), (weight - growths.get(&edge).unwrap_or(&0)).max(0));
            }
            let (node_map, visited_nodes_count) = dijkstra(&graph, (self.boundary_node_left as u32).into(), Some((self.boundary_node_right as u32).into()), |e| *e.weight());
            let result = node_map.get(&NodeIndex::new(self.boundary_node_right)).unwrap();
            preskill_time = now.elapsed().as_secs_f64();
            let mut num_nodes_with_not_zero_weight = 2; // 2 means two boundary nodes
            for node in 0..detection_events.len() {
                for &adj_node in self.link_list.get(&node).unwrap().iter() {
                    let edge = if node < adj_node { (node, adj_node) } else { (adj_node, node) };
                    if growths.get(&edge).unwrap_or(&0) < self.weights.get(&edge).unwrap() {
                        num_nodes_with_not_zero_weight += 1;
                        break;
                    }
                }
            }
            preskill_result = Some((*result, num_nodes_with_not_zero_weight, visited_nodes_count));
            //if *result < 0 {
            //    let peek_global_t = events.top().unwrap().0;
            //    println!("preskill result: {}", result);
            //    println!("peek_global_t: {}, event_cnt[peek_global_t]: {}, heap: {:?}", peek_global_t, events_cnt.get(&peek_global_t).unwrap(), events);
            //    println!("growths: {:?}", growths);
            //    println!("weights: {:?}", self.weights);
            //    assert!(*result >= 0);
            //}
            if debug {
                //let result = petgraph::algo::astar(&graph, (self.boundary_node_left as u32).into(), |finish| finish == (self.boundary_node_right as u32).into(), |e| *e.weight(), |_| 0);
                //if let Some((cost, path)) = result {
                //    println!("preskill(astar): {}", cost);
                //    println!("preskill path: {:?}", path);
                //}
            }
        }

        let mut bounded_dijkstra_result = None;
        let mut bounded_dijkstra_time = 0.;
        if use_bounded_dijkstra {
            let mut edges = Vec::with_capacity(self.weights.len());
            for &edge in self.weights.keys() {
                edges.push((edge.0 as u32, edge.1 as u32));
            }
            let mut graph = UnGraph::<u32, u64>::from_edges(&edges);
            let now = time::Instant::now();
            for (&edge, &weight) in &self.weights {
                graph.update_edge((edge.0 as u32).into(), (edge.1 as u32).into(), (weight - growths.get(&edge).unwrap_or(&0)).max(0));
            }
            let (node_map, visited_nodes_count) = bounded_dijkstra(&graph, (self.boundary_node_left as u32).into(), Some((self.boundary_node_right as u32).into()), |e| *e.weight(), additional_max_growth);
            let mut result = None;
            if let Some(node_map) = node_map {
                result = Some(*node_map.get(&NodeIndex::new(self.boundary_node_right)).unwrap());
            }
            bounded_dijkstra_time = now.elapsed().as_secs_f64();
            let mut num_nodes_with_not_zero_weight = 2; // 2 means two boundary nodes
            for node in 0..detection_events.len() {
                for &adj_node in self.link_list.get(&node).unwrap().iter() {
                    let edge = if node < adj_node { (node, adj_node) } else { (adj_node, node) };
                    if growths.get(&edge).unwrap_or(&0) < self.weights.get(&edge).unwrap() {
                        num_nodes_with_not_zero_weight += 1;
                        break;
                    }
                }
            }
            bounded_dijkstra_result = Some((result, num_nodes_with_not_zero_weight, visited_nodes_count));
        }

        let mut additional_growth_result = None;
        let mut additional_growth_time = 0.;
        let mut additional_det = HashSet::new();
        let mut cloned_det_to_de = HashMap::new();
        let mut additional_collisions: usize = 0;
        if debug {
            cloned_det_to_de = det_to_de.clone();
        }
        if additional_max_growth > 0 {
            //if debug {
            //    println!("ufroot_to_end_edges: {:?}", ufroot_to_end_edges);
            //}
            let now = time::Instant::now();
            let mut additional_growth_when_connecting_two_boundary_nodes = None;

            if uf_de.issame(self.boundary_node_left, self.boundary_node_right) {
                additional_growth_result = Some((0, 0, 0, 0));
            }
            else {
                let mut cluster_graph = HashMap::new();
                if debug {
                    //println!("uf_de.root(self.boundary_node_left): {}", uf_de.root(self.boundary_node_left));
                    //println!("uf_de.root(self.boundary_node_right): {}", uf_de.root(self.boundary_node_right));
                    //println!("uf_de.root(42): {}, uf_de.issame(42, 0): {}, uf_de.issame(self.boundary_node_right, 42): {}", uf_de.root(42), uf_de.issame(42, 0), uf_de.issame(self.boundary_node_right, 42));
                    //println!("uf_de.root(43): {}, uf_de.issame(43, 0): {}, uf_de.issame(self.boundary_node_right, 43): {}", uf_de.root(43), uf_de.issame(43, 0), uf_de.issame(self.boundary_node_right, 43));
                    //println!("uf_de.root(0): {}, uf_de.issame(self.boundary_node_right, 0): {}", uf_de.root(0), uf_de.issame(self.boundary_node_right, 0));
                    //println!("uf_de.issame(0, 65): {}, uf_de.issame(self.boundary_node_right, 65): {}", uf_de.issame(0, 65), uf_de.issame(self.boundary_node_right, 65));
                    //println!("uf_de.root(65): {}, uf_de.size(0): {}, uf_de.size(65): {}", uf_de.root(65), uf_de.size(0), uf_de.size(65));
                    //println!("uf_de.issame(0, 120): {}, uf_de.issame(0, 6): {}", uf_de.issame(0, 120), uf_de.issame(0, 6));
                    //println!("peek_global_t: {}, events: {:?}", prev_peek_global_t.unwrap(), events);
                    ////uf_de.unite(0, 65);
                    ////println!("uf_de.root(65): {}, uf_de.issame(0, 65): {}", uf_de.root(65), uf_de.issame(0, 65));
                }
                if self.boundary_node_left != uf_de.root(self.boundary_node_left) {
                    cluster_graph.insert((uf_de.root(self.boundary_node_left), self.boundary_node_left), 0); // uf_de.root(self.boundary_node) < self.boundary_node
                    let end_edges = ufroot_to_end_edges.get_mut(&uf_de.root(self.boundary_node_left)).unwrap();
                    for &adj_node in self.link_list.get(&self.boundary_node_left).unwrap().iter() {
                        let edge = (self.boundary_node_left, adj_node);
                        let cedge = common_edge(edge);
                        if growths.get(&cedge).unwrap_or(&0) >= self.weights.get(&cedge).unwrap() {
                            continue;
                        }
                        end_edges.insert(edge);
                    }
                    det_to_de.insert(self.boundary_node_left, uf_de.root(self.boundary_node_left));
                    if debug {
                        additional_det.insert(self.boundary_node_left);
                    }
                }
                else {
                    ufroot_to_end_edges.insert(self.boundary_node_left, self.link_list.get(&self.boundary_node_left).unwrap().iter().map(|&adj_node| (self.boundary_node_left, adj_node)).collect());
                    det_to_de.insert(self.boundary_node_left, self.boundary_node_left);
                    if debug {
                        additional_det.insert(self.boundary_node_left);
                    }
                }
                if self.boundary_node_right != uf_de.root(self.boundary_node_right) {
                    cluster_graph.insert((uf_de.root(self.boundary_node_right), self.boundary_node_right), 0);
                    let end_edges = ufroot_to_end_edges.get_mut(&uf_de.root(self.boundary_node_right)).unwrap();
                    for &adj_node in self.link_list.get(&self.boundary_node_right).unwrap().iter() {
                        let edge = (self.boundary_node_right, adj_node);
                        let cedge = common_edge(edge);
                        if growths.get(&cedge).unwrap_or(&0) >= self.weights.get(&cedge).unwrap() {
                            continue;
                        }
                        end_edges.insert(edge);
                    }
                    det_to_de.insert(self.boundary_node_right, uf_de.root(self.boundary_node_right));
                    if debug {
                        additional_det.insert(self.boundary_node_right);
                    }
                }
                else {
                    ufroot_to_end_edges.insert(self.boundary_node_right, self.link_list.get(&self.boundary_node_right).unwrap().iter().map(|&adj_node| (self.boundary_node_right, adj_node)).collect());
                    det_to_de.insert(self.boundary_node_right, self.boundary_node_right);
                    if debug {
                        additional_det.insert(self.boundary_node_right);
                    }
                }

                let mut uf_de_additional = UnionFind::new(detection_events.len() + 2);
                uf_de_additional.par = uf_de.par.clone();
                uf_de_additional.siz = uf_de.siz.clone();

                //let mut event_map: HashMap<(usize, usize), u64> = HashMap::new();
                //for (&de, &ref end_edges) in &ufroot_to_end_edges {
                //    for &edge in end_edges {
                //        let cedge = common_edge(edge);
                //        let &weight = self.weights.get(&cedge).unwrap();
                //        let &growth = growths.get(&cedge).unwrap_or(&0);
                //        let iedge = inverse_edge(edge);
                //        if let Some(&inverse_edge_value) = event_map.get(&iedge) {
                //            event_map.insert(edge, inverse_edge_value / 2);
                //            event_map.insert(iedge, inverse_edge_value / 2);
                //        }
                //        else {
                //            event_map.insert(edge, weight - growth);
                //        }
                //    }
                //}

                let mut event_map: HashMap<(usize, usize), HashMap<(usize, Option<usize>), u64, Hasher>, Hasher> = HashMap::default();
                for (&de, &ref end_edges) in &ufroot_to_end_edges {
                    for &edge in end_edges {
                        let de = uf_de.root(*det_to_de.get(&edge.0).unwrap());
                        let cedge = common_edge(edge);
                        let &weight = self.weights.get(&cedge).unwrap();
                        let &growth = growths.get(&cedge).unwrap_or(&0);
                        let iedge = inverse_edge(edge);
                        event_map.entry(edge).or_insert_with(|| HashMap::default()).insert((de, None), weight - growth);
                        if let Some(inverse_edge_dict) = event_map.get_mut(&iedge) {
                            let cloned_inverse_edge_dict = inverse_edge_dict.clone();
                            for (&(ide, end), &val) in &cloned_inverse_edge_dict {
                                if end.is_none() {
                                    inverse_edge_dict.insert((ide, Some(de)), val / 2);
                                }
                            }
                            let inverse_edge_dict = 0;
                            for (&(ide, end), &val) in &cloned_inverse_edge_dict {
                                if end.is_none() {
                                    event_map.entry(edge).or_insert_with(|| HashMap::default()).insert((de, Some(ide)), val / 2);
                                }
                            }
                        }
                    }
                }

                //let mut events_cnt = HashMap::new();
                let mut events = RadixHeapMap::new();
                for (&edge, &ref edge_dict) in &event_map {
                    for (&(de, ide), &t) in edge_dict {
                        events.push(Reverse(t), (edge, (de, ide)));
                    }
                }
                //for (&edge, &t) in &event_map {
                //    events.push(Reverse(t), edge);
                //    //*events_cnt.entry(t).or_insert(0) += 1;
                //}

                //let mut det_to_de_additional = HashMap::new();
                //for (&det, &de) in &det_to_de {
                //    det_to_de_additional.entry(det).or_insert(HashSet::new()).insert(uf_de.root(de));
                //}

                //if debug {
                //    println!("events: {:?}", events);
                //    println!("uf_de.root(92): {}, uf_de.root(99): {}", uf_de.root(92), uf_de.root(99));
                //}

                //for (&ufroot, &end_edges) in &ufroot_to_end_edges {
                //    if (let de = uf_de.root(ufroot)) != ufroot {
                //        let removed_end_edges = ufroot_to_end_edges.remove(&ufroot);
                //        ufroot_to_end_edges.get(&de).unwrap().extend(removed_end_edges);
                //    }
                //}

                loop {
                    if events.len() == 0 {
                        break;
                    }
                    //let peek_global_t = events.peek().unwrap().0.0;
                    events.constrain();
                    let peek_global_t = events.top().unwrap().0;
                    if peek_global_t > additional_max_growth / 2 {
                        break;
                    }
                    //let mut updated_roots = HashSet::new();
                    loop {
                        //println!("events: {:?}", events);
                        //println!("event_map: {:?}", event_map);
                        //println!("active_roots: {:?}", active_roots);
                        //println!("growths: {:?}", growths);
                        //println!("ufroot_to_end_edges: {:?}", ufroot_to_end_edges);
                        //println!("updated_roots: {:?}", updated_roots);
                        if events.len() == 0 {
                            break;
                        }
                        //if *events_cnt.get(&peek_global_t).unwrap() == 0 {
                        //    break;
                        //}

                        let peek_event = events.pop().unwrap();
                        let global_t = peek_event.0.0;
                        let peek_global_t = global_t;
                        if global_t > additional_max_growth / 2 {
                            break;
                        }
                        //assert!(global_t == peek_global_t);
                        let edge = peek_event.1.0;
                        let de = peek_event.1.1.0;
                        let ide = peek_event.1.1.1;
                        if debug {
                            additional_det.insert(edge.1);
                        }
                        //*events_cnt.entry(global_t).or_insert(0) -= 1;
                        //if !event_map.contains_key(&edge) {
                        //    continue;
                        //}
                        //if *event_map.get(&edge).unwrap() != global_t { // popped value is too old to use
                        //    continue;
                        //}

                        let cedge = common_edge(edge);
                        let iedge = inverse_edge(edge);
                        //growths.insert(cedge, *self.weights.get(&cedge).unwrap());
                        //event_map.remove(&edge);
                        //event_map.remove(&iedge);
                        //let des = det_to_de_additional.get(&edge.0).unwrap();
                        //let de = uf_de_additional.root(*det_to_de.get(&edge.0).unwrap());
                        //let de_original_uf = uf_de.root(*det_to_de.get(&edge.0).unwrap());
                        //let end_edges = ufroot_to_end_edges.get_mut(&de).unwrap();
                        //end_edges.remove(&edge);
                        //end_edges.remove(&inverse_edge(edge));

                        //if debug && edge == (99, 94) {
                        //    println!("global_t: {}", global_t);
                        //}

                        //if edge.1 == self.boundary_node_left || edge.1 == self.boundary_node_right {
                        //    //let boundary_root = uf_de_additional.root(edge.1);
                        //    //for &de in des {
                        //        uf_de_additional.unite(de, edge.1);
                        //    //}

                        //    //let current_root = uf_de_additional.root(edge.1);
                        //    //if boundary_root == current_root && de != current_root {
                        //    //    let removed_end_edges = ufroot_to_end_edges.remove(&de).unwrap();
                        //    //    let end_edges = ufroot_to_end_edges.get_mut(&current_root).unwrap();
                        //    //    end_edges.extend(removed_end_edges);
                        //    //}
                        //    //else if boundary_root != edge.1 && boundary_root != current_root {
                        //    //    let removed_end_edges = ufroot_to_end_edges.remove(&boundary_root).unwrap();
                        //    //    let end_edges = ufroot_to_end_edges.get_mut(&de).unwrap();
                        //    //    end_edges.extend(removed_end_edges);
                        //    //}

                        //    //for &de in des {
                        //        if !cluster_graph.contains_key(&common_edge((de, edge.1))) {
                        //            cluster_graph.insert(common_edge((de, edge.1)), global_t);
                        //        }
                        //    //}

                        //    //updated_roots.insert(current_root);
                        //}
                        /*else*/ if let Some(adj_de) = ide {
                            if !cluster_graph.contains_key(&common_edge((de, adj_de))) {
                                cluster_graph.insert(common_edge((de, adj_de)), global_t * 2);
                                if debug {
                                    additional_collisions += 1;
                                }
                            }
                            uf_de_additional.unite(de, adj_de);
                        }
                        //else if let Some(adj_des) = det_to_de_additional.get(&edge.1) {
                        //    //if debug && edge.1 == 94 {
                        //    //    println!("edge: {}, adj_de: {}, global_t: {}, uf_de.root(de): {}, uf_de.root(adj_de): {}", edge, adj_de, global_t, uf_de.root(de), uf_de.root(adj_de));
                        //    //}
                        //    for &de in des {
                        //        for &adj_de in adj_des {
                        //            if de != adj_de {
                        //                if !cluster_graph.contains_key(&common_edge((de, adj_de))) {
                        //                    cluster_graph.insert(common_edge((de, adj_de)), global_t * 2);
                        //                }
                        //                uf_de_additional.unite(de, adj_de);
                        //            }
                        //        }
                        //    }
                        //    let des = det_to_de_additional.get(&edge.0).unwrap().clone();
                        //    let adj_des = det_to_de_additional.get_mut(&edge.1).unwrap();
                        //    for de in des {
                        //        adj_des.insert(de);
                        //    }
                        //    //if !uf_de.issame(de_original_uf, adj_de) {
                        //    //    let adj_de_original_uf = uf_de.root(adj_de);
                        //    //    let adj_de = uf_de_additional.root(adj_de);
                        //    //    if !cluster_graph.contains_key(&common_edge((de_original_uf, adj_de_original_uf))) {
                        //    //        cluster_graph.insert(common_edge((de_original_uf, adj_de_original_uf)), global_t * 2);
                        //    //    }

                        //    //    if !uf_de_additional.issame(de, adj_de) {
                        //    //        uf_de_additional.unite(de, adj_de);

                        //    //        //let new_root = uf_de_additional.root(de);
                        //    //        //if de == new_root {
                        //    //        //    let adj_end_edges = ufroot_to_end_edges.remove(&adj_de).unwrap();
                        //    //        //    let end_edges = ufroot_to_end_edges.get_mut(&de).unwrap();
                        //    //        //    end_edges.extend(adj_end_edges);
                        //    //        //    end_edges.remove(&iedge);
                        //    //        //}
                        //    //        //else {
                        //    //        //    let end_edges = ufroot_to_end_edges.remove(&de).unwrap();
                        //    //        //    let adj_end_edges = ufroot_to_end_edges.get_mut(&adj_de).unwrap();
                        //    //        //    adj_end_edges.extend(end_edges.into_iter());
                        //    //        //    adj_end_edges.remove(&iedge);
                        //    //        //}
                        //    //        //updated_roots.insert(new_root);
                        //    //    }
                        //    //}
                        //}
                        else {
                            //let des = det_to_de_additional.get(&edge.0).unwrap().clone();
                            //for de in des {
                            //    det_to_de_additional.entry(edge.1).or_insert(HashSet::new()).insert(de);
                            //}
                            //det_to_de.insert(edge.1, de_original_uf);
                            for &adj_node in self.link_list.get(&edge.1).unwrap().iter() {
                                let end_edge = (edge.1, adj_node);
                                if end_edge == edge || inverse_edge(end_edge) == edge {
                                    continue;
                                }

                                let end_cedge = common_edge(end_edge);
                                let end_iedge = inverse_edge(end_edge);

                                let weight = self.weights.get(&end_cedge).unwrap();
                                let growth = growths.get(&end_cedge).unwrap_or(&0);
                                if growth >= weight {
                                    continue;
                                }

                                //end_edges.insert(end_edge);

                                if let Some(inverse_edge_dict) = event_map.get_mut(&end_iedge) {
                                    let cloned_inverse_edge_dict = inverse_edge_dict.clone();
                                    //if debug && edge == (202, 250) {
                                    //    println!("(202, 250), iedge_dict: {:?}", cloned_inverse_edge_dict);
                                    //}
                                    for (&(ide, end), &inverse_edge_value) in &cloned_inverse_edge_dict {
                                        if end.is_none() && inverse_edge_value >= peek_global_t {
                                            let updated_growth = weight - (inverse_edge_value - peek_global_t);
                                            let val = peek_global_t + (weight - updated_growth) / 2;
                                            inverse_edge_dict.insert((ide, Some(de)), val);
                                            events.push(Reverse(val), (end_iedge, (ide, Some(de))));
                                            //if debug && edge == (202, 250) {
                                            //    println!("(202, 250), iedge_dict: {:?}", inverse_edge_dict);
                                            //}
                                        }
                                    }
                                    let inverse_edge_dict = 0;
                                    for (&(ide, end), &inverse_edge_value) in &cloned_inverse_edge_dict {
                                        if end.is_none() && inverse_edge_value >= peek_global_t {
                                            let updated_growth = weight - (inverse_edge_value - peek_global_t);
                                            let val = peek_global_t + (weight - updated_growth) / 2;
                                            event_map.entry(end_edge).or_insert_with(|| HashMap::default()).insert((de, Some(ide)), val);
                                            events.push(Reverse(val), (end_edge, (de, Some(ide))));
                                        }
                                    }
                                }
                                //if !event_map.contains_key(&end_edge) {
                                if !event_map.contains_key(&end_edge) || !event_map.get(&end_edge).unwrap().contains_key(&(de, None)) {
                                    let val = peek_global_t + weight - growth;
                                    event_map.entry(end_edge).or_insert_with(|| HashMap::default()).insert((de, None), val);
                                    events.push(Reverse(val), (end_edge, (de, None)));
                                }
                                //}

                                //if let Some(edge_value) = event_map.get(&end_edge) {
                                //    continue;
                                //}
                                //else if let Some(inverse_edge_value) = event_map.get(&end_iedge) {
                                //    //println!("end_edge: {:?} and end_iedge: {:?} is updated", end_edge, end_iedge);
                                //    let updated_growth = weight - (inverse_edge_value - peek_global_t);
                                //    growths.insert(end_cedge, updated_growth);
                                //    let val = peek_global_t + (weight - updated_growth) / 2;
                                //    event_map.insert(end_edge, val);
                                //    event_map.insert(end_iedge, val);
                                //    assert!(val >= peek_global_t);
                                //    //println!("hoge");
                                //    //println!("val: {}, peek_global_t: {}, top: {}, global_t: {}", val, peek_global_t, events.top().unwrap().0, global_t);
                                //    events.push(Reverse(val), end_edge);
                                //    events.push(Reverse(val), end_iedge);
                                //    //println!("piyo");
                                //    //*events_cnt.entry(val).or_insert(0) += 2;
                                //}
                                //else {
                                //    //println!("end_edge: {:?} is updated", end_edge);
                                //    let val = peek_global_t + weight - growth;
                                //    event_map.insert(end_edge, val);
                                //    //events.push(Reverse((val, end_edge)));
                                //    assert!(val >= peek_global_t);
                                //    events.push(Reverse(val), end_edge);
                                //    //*events_cnt.entry(val).or_insert(0) += 1;
                                //}
                            }
                        }
                    }
                    //if active_roots.len() == 0 { // to be simple for preskill and additional growth, this small optimization is removed
                    //    break;
                    //}

                    //println!("updated_roots before collect: {:?}", updated_roots);
                    //let updated_roots: HashSet<_> = updated_roots.into_iter().map(|de| uf_de.root(de)).collect();
                    //println!("updated_roots after collect: {:?}", updated_roots);
                    //assert!(peek_global_t )

                    //let debug_now = time::Instant::now();
                    //for &de in &updated_roots {
                    //    if let Some(end_edges) = ufroot_to_end_edges.get(&de) {
                    //        for &end_edge in end_edges {
                    //            let end_cedge = common_edge(end_edge);
                    //            let weight = self.weights.get(&end_cedge).unwrap();
                    //            let end_iedge = inverse_edge(end_edge);

                    //            if !event_map.contains_key(&end_edge) {
                    //                if let Some(inverse_edge_value) = event_map.get(&end_iedge) {
                    //                    let updated_growth = weight - (inverse_edge_value - peek_global_t);
                    //                    //println!("2x; end_edge: {:?}, inverse_edge_value: {}, peek_global_t: {}, weight: {}", end_edge, inverse_edge_value, peek_global_t, weight);
                    //                    growths.insert(end_cedge, updated_growth);
                    //                    let val = peek_global_t + (weight - updated_growth) / 2;
                    //                    event_map.insert(end_edge, val);
                    //                    event_map.insert(end_iedge, val);
                    //                    //events.push(Reverse((val, end_edge)));
                    //                    //events.push(Reverse((val, end_iedge)));
                    //                    assert!(val >= peek_global_t);
                    //                    //println!("val: {}, peek_global_t: {}, top: {}", val, peek_global_t, events.top().unwrap().0);
                    //                    events.push(Reverse(val), end_edge);
                    //                    events.push(Reverse(val), end_iedge);
                    //                    //println!("piyo");
                    //                    *events_cnt.entry(val).or_insert(0) += 2;
                    //                }
                    //                else {
                    //                    let growth = growths.get(&end_cedge).unwrap_or(&0);
                    //                    //println!("1x; end_edge: {:?}, growth: {}, peek_global_t: {}, weight: {}", end_edge, growth, peek_global_t, weight);
                    //                    let val = peek_global_t + weight - growth;
                    //                    event_map.insert(end_edge, val);
                    //                    //events.push(Reverse((val, end_edge)));
                    //                    assert!(val >= peek_global_t);
                    //                    events.push(Reverse(val), end_edge);
                    //                    *events_cnt.entry(val).or_insert(0) += 1;
                    //                }
                    //            }
                    //            // if edge's event_map exists, do nothing here.
                    //        }
                    //    }
                    //}
                    //debug_time += debug_now.elapsed().as_secs_f64();
                    if additional_growth_when_connecting_two_boundary_nodes.is_none() && uf_de_additional.issame(self.boundary_node_left, self.boundary_node_right) {
                        additional_growth_when_connecting_two_boundary_nodes = Some(peek_global_t);
                    }
                }

                if debug {
                    //println!("cluster_graph: {:?}", cluster_graph);
                    //println!("peek_global_t: {}", events.top().unwrap().0);
                    //println!("events: {:?}", events);
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
                    let mut graph = UnGraph::<u32, u64>::from_edges(&edges);
                    for (&edge, &weight) in &cluster_graph {
                        graph.update_edge((edge.0 as u32).into(), (edge.1 as u32).into(), weight);
                    }
                    let (node_map, visited_nodes_count) = dijkstra(&graph, (self.boundary_node_left as u32).into(), Some((self.boundary_node_right as u32).into()), |e| *e.weight());
                    //println!("{:?}", node_map);
                    let result = node_map.get(&NodeIndex::new(self.boundary_node_right)).unwrap();
                    additional_growth_result = Some((additional_growth_when_connecting_two_boundary_nodes.unwrap(), *result, nodes.len(), visited_nodes_count));//graph.node_count())); // graph.node_count() seems broken, and it returns the maximum node index + 1.
                    if debug {
                        //println!("det_to_de_additional: {:?}", det_to_de_additional);
                        //println!("{:?}", cluster_graph);
                        //let result = petgraph::algo::astar(&graph, (self.boundary_node_left as u32).into(), |finish| finish == (self.boundary_node_right as u32).into(), |e| *e.weight(), |_| 0);
                        //if let Some((cost, path)) = result {
                        //    println!("pc(astar): {}", cost);
                        //    println!("pc path: {:?}", path);
                        //}
                    }
                }

                additional_growth_time = now.elapsed().as_secs_f64();
            }
        }

        ((uf_de.size(self.boundary_node_left)-1) % 2 == 1,
            preskill_result,
            bounded_dijkstra_result,
            additional_growth_result,
            (uf_time, preskill_time, bounded_dijkstra_time, additional_growth_time, debug_time),
            if debug { Some(growths.iter().filter(|(&key, &value)| value >= *self.weights.get(&key).unwrap()).map(|(&key, &value)| key).collect()) } else { None },
            if debug { Some(growths) } else { None },
            if debug { Some(cloned_det_to_de) } else { None },
            if debug { Some(additional_det) } else { None },
            if debug { Some(additional_collisions) } else { None },
            prev_peek_global_t,
            )
    }
}

fn common_edge(edge: (usize, usize)) -> (usize, usize) {
    if edge.0 < edge.1 { (edge.0, edge.1) } else { (edge.1, edge.0) }
}

fn inverse_edge(edge: (usize, usize)) -> (usize, usize) {
    (edge.1, edge.0)
}

//trait Peek<K, V> {
//    fn peek(&mut self) -> Option<&(K, V)>;
//}
//
//impl<K, V> Peek<K, V> for RadixHeapMap<K, V> 
//where
//    K: radix_heap::Radix + Ord + std::marker::Copy,
//    V: std::marker::Copy
//{
//    fn peek(&mut self) -> Option<&(K, V)> {
//        //if self.buckets[0].is_empty() {
//        //    self.constrain();
//        //}
//        //self.buckets[0].last()
//        
//        if let Some(popped_bucket) = self.pop() {
//            //self.push(popped_bucket.0, popped_bucket.1);
//            //return Some(&popped_bucket);
//            let copied_bucket = popped_bucket.clone();
//            self.push(copied_bucket.0, copied_bucket.1);
//            return Some(&copied_bucket);
//        }
//        None
//    }
//}
