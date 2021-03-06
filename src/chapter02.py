from typing import Any, List, Mapping, Union
import functools
import operator

import numpy as np
from scipy.sparse import csr_matrix as sparse_matrix


def assign_element_indices(mesh: List[List[Any]]) -> Tuple[Mapping[Tuple[Any], int]]:
	"""
	we will assume that the mesh is represented as a list of lists – convert to tuple for hashing
	"""
	vertices = [tuple(x) for x in mesh if len(x) == 1]
	edges = [tuple(x) for x in mesh if len(x) == 2]
	faces = [tuple(x) for x in mesh if len(x) == 3]

	vertex_mapping = {x: i for i, x in enumerate(vertices)}
	edge_mapping = {x: i for i, x in enumerate(edges)}
	face_mapping = {x: i for i, x in enumerate(faces)}

	return vertex_mapping, edge_mapping, face_mapping


def build_vertex_edge_adjacency_matrix(mesh: List[List[Any]]) -> sparse_matrix:
	vertex_mapping, edge_mapping, face_mapping = assign_element_indices(mesh)
	vertex_edge_adjacency_matrix = np.zeros((len(vertex_mapping.keys()), len(edge_mapping.keys())))

	for v in vertex_mapping.keys():
		for e in edge_mapping.keys():
			if v[0] in e:
				vertex_edge_adjacency_matrix[vertex_mapping[v], edge_mapping[e]] = 1
			else:
				vertex_edge_adjacency_matrix[vertex_mapping[v], edge_mapping[e]] = 0

	sparse_adjacency_matrix = sparse_matrix(vertex_edge_adjacency_matrix)
	return sparse_adjacency_matrix


def build_edge_face_adjacency_matrix(mesh: List[List[Any]]) -> sparse_matrix:
	vertex_mapping, edge_mapping, face_mapping = assign_element_indices(mesh)
	edge_face_adjacency_matrix = np.zeros((len(edge_mapping.keys()), len(face_mapping.keys())))

	for e in edge_mapping.keys():
		for f in face_mapping.keys():
			if e[0] in f and e[1] in f:
				edge_face_adjacency_matrix[edge_mapping[e], face_mapping[f]] = 1
			else:
				edge_face_adjacency_matrix[edge_mapping[e], face_mapping[f]] = 0

	sparse_adjacency_matrix = sparse_matrix(edge_face_adjacency_matrix)
	return sparse_adjacency_matrix


# thanks to SWE stackexchange user @Shreyas for this helper function – i modified it to parametrize length
def flatten(any_list: List[Any], flatten_depth=1) -> List[Any]: 
	return flatten(any_list[0]) + (flatten(any_list[1:]) if len(any_list) > flatten_depth else []) if type(any_list) is list else [any_list]


def build_vertex_vector(simplicial_set: List[List[Any]], subset: List[List[Any]]) -> np.array:
	for s in subset:
		assert s in simplicial_set

	# we expect that elements of simplicial_set and subset will be iterables
	vertex_list = [s[0] for s in simplicial_set if len(s) == 1]
	vertex_mapping = {v: i for i, v in enumerate(vertex_list)}
	vertex_dict = {v: 0 for v in vertex_list}
	
	# i am assuming that we are counting vertices that take part in edges/faces, even if the vertices are not explicitly subsets themselves
	# ... if not, then this is wrong. but the code for that case is more straightforward.
	subset = flatten(subset)
	subset = set(subset)

	for s in subset:
		vertex_dict[s] = 1

	# for interoperability, need to standardize by sorting vertex encodings (and downstream, this mapping needs to be shared)
	vertex_dict = {vertex_mapping[vertex]: value for vertex, value in vertex_dict.items()}
	vertex_dict = dict(sorted(vertex_dict.items()))

	vertex_vector = np.array(vertex_dict.values())
	return vertex_vector


def build_edge_vector(simplicial_set: List[List[Any]], subset: List[List[Any]]) -> np.array:
	for s in subset:
		assert s in simplicial_set

	edge_list = [tuple(s) for s in simplicial_set if len(s) == 2]
	edge_mapping = {e: i for i, e in enumerate(edge_list)}
	edge_dict = {e: 0 for e in edge_list}

	for s in subset:
		if len(s) == 2:
			edge_dict[s] = 1

	# for interoperability, need to standardize by sorting edge encodings (and downstream, this mapping needs to be shared)
	edge_dict = {edge_mapping[edge], value for edge, value in edge_dict.items()}
	edge_dict = dict(sorted(edge_dict.items()))

	edge_vector = np.array(edge_dict.values())
	return edge_vector


def build_face_vector(simplicial_set: List[List[Any]], subset: List[List[Any]]) -> np.array:
	for s in subset:
		assert s in simplicial_set

	face_list = [tuple(s) for s in simplicial_set if len(s) == 3]
	face_mapping = {f: i for i, f in enumerate(face_list)}
	face_dict = {f: 0 for f in face_list}

	for s in subset:
		if len(s) == 3:
			face_dict[s] = 1

	face_dict = {face_mapping[face], value for face, value in face_dict.items()}
	face_dict = dict(sorted(face_dict.items()))

	face_vector = np.array(face_dict.values())
	return face_vector


def build_simplicial_vector(simplicial_set: List[List[Any]], subset: List[List[Any]], simplex_type: Union[str, int]) -> np.array:
	# you can now iterate over higher-order simplex-types 
	for s in subset:
		assert s in simplicial_set

	simplex_type_to_length = {
		'vertex': 1,
		'edge': 2,
		'face': 3,
		'triangle': 3
	}

	if isinstance(simplex_type, int):
		simplex_length = simplex_type
	else:
		simplex_length = simplex_type_to_length[simplex_type]

	simplex_list = [tuple(s) for s in simplicial_set if len(s) == simplex_length]
	simplex_mapping = {s: i for i, s in enumerate(simplex_list)}
	simplex_dict = {s: 0 for s in simplex_list}

	for s in subset:
		if len(s) == simplex_length:
			simplex_dict = 1

	simplex_dict = {simplex_mapping[simplex], value for simplex, value in simplex_dict.items()}
	simplex_dict = dict(sorted(simplex_dict.items()))

	simplex_vector = np.array(simplex_dict.values())
	return simplex_vector





