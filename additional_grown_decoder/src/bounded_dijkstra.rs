use alloc::collections::BinaryHeap;
use core::hash::Hash;

use hashbrown::hash_map::{
    Entry::{Occupied, Vacant},
    HashMap,
};

use petgraph::algo::Measure;
use crate::scored::MinScored;
use petgraph::visit::{EdgeRef, IntoEdges, VisitMap, Visitable};


pub fn bounded_dijkstra<G, F, K>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    mut edge_cost: F,
    bound: K,
) -> (Option<HashMap<G::NodeId, K>>, usize)
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> K,
    K: Measure + Copy,
{
    let mut visited_nodes_count: usize = 0;
    let mut visited = graph.visit_map();
    let mut scores = HashMap::new();
    //let mut predecessor = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.insert(start, zero_score);
    visit_next.push(MinScored(zero_score, start));
    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        visited_nodes_count += 1;
        if goal.as_ref() == Some(&node) {
            break;
        }
        if node_score > bound {
            return (None, visited_nodes_count);
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            let next_score = node_score + edge_cost(edge);
            match scores.entry(next) {
                Occupied(ent) => {
                    if next_score < *ent.get() {
                        *ent.into_mut() = next_score;
                        visit_next.push(MinScored(next_score, next));
                        //predecessor.insert(next.clone(), node.clone());
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                    //predecessor.insert(next.clone(), node.clone());
                }
            }
        }
        visited.visit(node);
    }
    (Some(scores), visited_nodes_count)
}