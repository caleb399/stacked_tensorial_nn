import pydot
import re
from keras.models import Model
from keras.layers import Layer, InputLayer
from pygments.lexers import graphviz


# May be necessary to manually add Graphviz to PATH, e.g.
# import os
# os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

def visualize_model(model, layer_labels = None, layer_colors = None, groupings = None, exclude_input_layer = False,
					verbose = False, output_filename = 'model_graph.png'):
	"""
	Creates a visual graph of a keras model. There is an option to group certain layers into subgraphs
	(argument 'groupings').

	Args:
		model: A Keras Model instance
		layer_labels (optional): List of labels for each layer. Defaults to layer names.
		layer_colors (optional): List of colors for each layer. Defaults to white for all layers.
		groupings (optional): Dictionary specifying groups of layers. Each key is a group name,
							  and its value is a list of layer names belonging to that group.
		exclude_input_layer (optional): Boolean indicating whether to exclude the input layer from the graph.
		verbose (boolean, optional): Whether to print verbose output. Defaults to False.
		output_filename (optional): name of the output file for saving the generated graph.

	Output:
		Image file with name 'output_filename'.
	"""
	if not isinstance(model, Model):
		raise ValueError("model should be a Keras model instance")
	num_layers = len(model.layers)

	# Default labels and colors if not provided
	if not layer_labels:
		layer_labels = [layer.name for layer in model.layers]
	if not layer_colors:
		default_color = 'white'
		layer_colors = [default_color] * num_layers

	# Create a directed graph
	graph = pydot.Dot(graph_type = 'digraph', rankdir = 'LR')

	# Create nodes for each layer and add to subgraphs if specified
	subgraphs = {}
	layer_id_map = {}
	for i, layer in enumerate(model.layers):
		# Exclude the input layer if specified
		if exclude_input_layer and isinstance(layer, InputLayer):
			continue

		# Create a node for the layer
		layer_id = str(id(layer))
		layer_id_map[layer] = layer_id
		label = layer_labels[i]
		color = layer_colors[i]

		node = pydot.Node(layer_id, label = label, style = 'filled', fillcolor = color, shape = 'box')

		# Check for groupings and add the node to the appropriate subgraph or main graph
		group_name = None
		if groupings:
			for group, members in groupings.items():
				if layer.name in members:
					group_name = group
					break

		if group_name:
			if group_name not in subgraphs:
				subgraph = pydot.Cluster(group_name, label = group_name, style = 'dashed', fontsize = 24)
				subgraphs[group_name] = subgraph
			subgraphs[group_name].add_node(node)
		else:
			graph.add_node(node)

	# Add subgraphs to the main graph
	for subgraph in subgraphs.values():
		graph.add_subgraph(subgraph)

	# Add edges based on layer connections
	for layer in model.layers:
		if exclude_input_layer and isinstance(layer, InputLayer):
			continue
		# Handle custom or non-standard layers
		if hasattr(layer, '_inbound_nodes'):
			inbound_nodes = layer._inbound_nodes
		else:
			# If the layer doesn't have '_inbound_nodes', skip edge creation
			continue

		inbound_layers = []
		for inbound_node in inbound_nodes:
			inbound_layers = inbound_node.inbound_layers
			if not isinstance(inbound_layers, list):
				inbound_layers = [inbound_layers]

		for inbound_node in inbound_nodes:
			for inbound_layer in inbound_layers:
				if isinstance(inbound_layer, Layer) and inbound_layer in layer_id_map:
					src_id = layer_id_map[inbound_layer]
					dest_id = layer_id_map[layer]
					if (re.search('sequential', inbound_layer.name, flags = re.IGNORECASE) or
							re.search(r'operators__.getitem_[0-9]+$', inbound_layer.name, flags = re.IGNORECASE)):
						graph.add_edge(pydot.Edge(src_id, dest_id, style = 'invis'))
					else:
						graph.add_edge(pydot.Edge(src_id, dest_id))
					if verbose:
						print(f"Added edge from {inbound_layer.name} to {layer.name}")

	graph.set_graph_defaults(sep = '+125,125')
	try:
		graph.write_png(output_filename)
	except FileNotFoundError as e:
		print(f'\nFailed to create network visualization using pydot and graphviz. Pleasure ensure that '
			  'the output filename is valid, and graphviz is installed and included in the system PATH variable. '
			  f'Original error: {e}')
	except Exception as e:
		print(f'\nFailed to create network visualization using pydot and graphviz. Original error: {e}')
	else:
		print(f'Model visualization saved to {output_filename}')
