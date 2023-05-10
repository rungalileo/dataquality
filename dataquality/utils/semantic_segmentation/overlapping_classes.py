import json
from collections import Counter
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import torch
from networkx.classes.graph import Graph

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import GALILEO_DEFAULT_RESULT_BUCKET_NAME
from dataquality.schemas.semantic_segmentation import Community

COMM_GRAPH_NAME = "community_graph"
COMMUNITY_SCORES_NAME = "community_scores"
object_store = ObjectStore()


def convert_to_undirected(graph: nx.DiGraph) -> Graph:
    Ugraph = graph.to_undirected()
    for node in graph:
        for ngbr in nx.neighbors(graph, node):
            if node != ngbr and node in nx.neighbors(
                graph, ngbr
            ):  # Don't double count self loops!
                Ugraph.edges[node, ngbr]["weight"] = (
                    graph.edges[node, ngbr]["weight"]
                    + graph.edges[ngbr, node]["weight"]
                )
    return Ugraph


def add_batch_to_graph(
    graph: nx.DiGraph, probs: np.ndarray, sample_labels: np.ndarray, top_k: int
) -> nx.DiGraph:
    """Creates (or continues to populate) the graph based on the batch of given probs

    For each example, add a weighted edge between the GT label node
    and the kth predicted class node with weight = predicted probability.
    Conceptually we reduce the mutli-graph to our final graph by summing the weights of
    all the edges between two node classes and creating a single weighted edge.

    Only add weight based on the top_k predicted classes. Note: if top_k = -1
    then we include all classes.

    :param graph: The DiGraph to be added to
    :param probs: The probability matrix for each sample in the batch
    :param sample_labels: The label idx for each samplee in hte batch
    :param top_k: How many top prediction classes to look at
    """
    for prob, src_label in zip(probs, sample_labels):
        sorted_probs = np.argsort(prob)

        for i in range(1, top_k + 1):
            pred_label = sorted_probs[-i]
            new_edge_weight = prob[pred_label]
            # Check if an edge exists
            if graph.has_edge(src_label, pred_label):
                graph[src_label][pred_label]["weight"] += new_edge_weight
            else:
                graph.add_edge(src_label, pred_label, weight=new_edge_weight)
    return graph


def normalize_graph(graph: nx.DiGraph, samples_per_class: Counter) -> nx.DiGraph:
    """Normalize the graph weights by the ratio of samples in the community

    :param graph: The DirectGraph to be normalized
    :param sample_labels: The label idx for each sample in the datasett
    """
    # For each node, convert weight of outgoing edge as [weight / (num_samples / total)]
    total_samples = sum(samples_per_class.values())
    for node in graph.nodes:
        normalize_factor = samples_per_class[node] / total_samples
        for neighbor in graph[node]:
            graph[node][neighbor]["weight"] = (
                graph[node][neighbor]["weight"] / normalize_factor
            )

    return graph


def upload_graph_obj(graph: Graph, project_run: str, split: str) -> None:
    """We save the graph in object store because we may want to provide pruning

    and further filtering in the API in the future (a threshold slider).
    """
    object_store = ObjectStore()
    with NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
        nx.write_edgelist(graph, f.name)

    graph_object_path = f"{project_run}/{split}/{COMM_GRAPH_NAME}"
    object_store.create_object(
        object_name=graph_object_path,
        file_path=f.name,
        content_type="text/plain",
        bucket_name=GALILEO_DEFAULT_RESULT_BUCKET_NAME,
        progress=False,
    )


def compute_louvain_communities(
    graph: Graph, resolution: int, random_state: int
) -> List[List[int]]:
    optimal_partition = nx_comm.louvain_communities(
        graph, seed=random_state, resolution=resolution, weight="weight"
    )
    # We don't want communities with 1 class
    optimal_partition = [part for part in optimal_partition if len(part) > 1]
    return list(map(list, optimal_partition))


def compute_community_probability_mass(
    communities: List[List[int]],
    probability_queue: torch.Tensor,
    gt_queue: torch.Tensor,
) -> Tuple[List[float], List[int]]:
    """Compute the probability mass captured by the community!

    For each sample in the given class, compute the probability mass (sum) of the
    non ground truth classes that exist in the community of the ground truth class

    Ex:
        Sample 1, GT = 4, community = [1, 3, 4]
            prob vector: [0.1, 0.05, 0.05, 0.25, 0.55]
            Probability mass/sum = 0.05+0.25 = 0.3
        Sample 2, GT = 1, community = [1, 3, 4]
            prob vector: [0.1, 0.6, 0.1, 0.1, 0.1]
            Probability mass/sum = 0.1+0.1 = 0.2
        Score = (0.3 + 0.2) / 2 = 0.25

    Returns a list of approximate probability masses (the score) for each community
    based on our queue, and a list of the sizes (num_samples) of each community.

    num_samples is simply the total number of samples for the classes in the dataset.
    So if the dataset has classes 1,2,3,4,5,6 and a community has classes [2,4]
    then num_samples is just the count of samples in the dataset whose GT are 2
    or 4. `np.where((labels==2) | (labels==4))` or `df["gold"].isin([2,4])`
    """
    probability_shape = probability_queue.shape
    np_probability_queue = probability_queue.view(-1, probability_shape[-1]).numpy()
    np_gt_queue = gt_queue.view(-1, 1).numpy()
    prob_masses = []
    comm_sizes = []
    for comm_labels in communities:
        gt_mask = (np_gt_queue == comm_labels).any(axis=1)
        comm_probs = np_probability_queue[gt_mask]

        gt_prob = np.take(comm_probs, np_gt_queue[gt_mask].reshape(-1).astype(int))
        comm_probs = comm_probs[:, np.array(comm_labels)]
        comm_probs = comm_probs.sum(axis=1) - gt_prob
        avg_prob_mass = comm_probs.mean().round(3)

        prob_masses.append(avg_prob_mass)
        comm_sizes.append(len(comm_probs))

    return prob_masses, comm_sizes


def save_community_scores(
    communities: List[List[int]],
    comm_scores: List[float],
    comm_sizes: List[int],
    project_run: str,
    split: str,
) -> None:
    """Stores community scores for run/split in DB

    Each split has a set of communities based on the class confusion of its samples
    The main metric is the number of communities, and the `extra` contains their
    scores as well as the classes within each community, because a community
    is defined as the classes within it. (Class overlap is at the class level, not the
    sample level)
    """
    # Return the string labels to the UI, not the label idxs
    comms = [
        Community(score=score, num_samples=size, labels=[idx for idx in comm]).dict()
        for score, size, comm in zip(comm_scores, comm_sizes, communities)
    ]

    with NamedTemporaryFile(mode="w+", delete=False) as f:
        json.dump(comms, f)

    object_name = f"{project_run}/{split}/{COMMUNITY_SCORES_NAME}"
    object_store.create_object(
        object_name=object_name,
        file_path=f.name,
        content_type="application/json",
        progress=False,
        bucket_name=GALILEO_DEFAULT_RESULT_BUCKET_NAME,
    )
