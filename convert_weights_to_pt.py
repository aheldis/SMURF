from argparse import ArgumentParser

import torch
from tensorflow.python.tools.inspect_checkpoint import py_checkpoint_reader

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--tf_checkpoint", required=True, type=str, help="Path to original Tensorflow checkpoint.")
parser.add_argument(
    "--pt_checkpoint_path", default="raft_smurf.pt", type=str, help="Path of converted PyTorch checkpoint."
)

# Get arguments
args = parser.parse_args()

# TF keys to PyTorch
TFtoPT = {
    # Context encoder
    "feature_model/cnet/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.conv1.weight",
    "feature_model/cnet/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.conv1.bias",
    "feature_model/cnet/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.conv2.weight",
    "feature_model/cnet/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.conv2.bias",
    "feature_model/cnet/layer1/layer_with_weights-0/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.0.conv1.weight",
    "feature_model/cnet/layer1/layer_with_weights-0/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.0.conv1.bias",
    "feature_model/cnet/layer1/layer_with_weights-0/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.0.conv2.weight",
    "feature_model/cnet/layer1/layer_with_weights-0/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.0.conv2.bias",
    "feature_model/cnet/layer1/layer_with_weights-0/norm1/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.0.norm1.weight",
    "feature_model/cnet/layer1/layer_with_weights-0/norm1/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.0.norm1.bias",
    "feature_model/cnet/layer1/layer_with_weights-0/norm2/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.0.norm2.weight",
    "feature_model/cnet/layer1/layer_with_weights-0/norm2/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.0.norm2.bias",
    # Repeat similar entries for other layers in layer1, layer2, and layer3
    "feature_model/cnet/layer1/layer_with_weights-1/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.1.conv1.weight",
    "feature_model/cnet/layer1/layer_with_weights-1/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.1.conv1.bias",
    "feature_model/cnet/layer1/layer_with_weights-1/conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.1.conv2.weight",
    "feature_model/cnet/layer1/layer_with_weights-1/conv2/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.1.conv2.bias",
    "feature_model/cnet/layer1/layer_with_weights-1/norm1/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.1.norm1.weight",
    "feature_model/cnet/layer1/layer_with_weights-1/norm1/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.1.norm1.bias",
    "feature_model/cnet/layer1/layer_with_weights-1/norm2/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.1.norm2.weight",
    "feature_model/cnet/layer1/layer_with_weights-1/norm2/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.cnet.layer1.1.norm2.bias",

    # Continue for other layers, ensuring to add norm layers where applicable
    # Feature encoder
    "feature_model/fnet/conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.fnet.conv1.weight",
    "feature_model/fnet/conv1/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.fnet.conv1.bias",
    # Add missing normalization layers similarly
    # Update block
    "flow_model/update_block/encoder/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.update_block.encoder.conv.weight",
    "flow_model/update_block/encoder/conv/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.update_block.encoder.conv.bias",
    # Add missing normalization layers similarly

    # Mask predictor
    "flow_model/update_block/mask/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.update_block.mask.0.weight",
    "flow_model/update_block/mask/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.update_block.mask.0.bias",
    "flow_model/update_block/mask/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE": "module.update_block.mask.2.weight",
    "flow_model/update_block/mask/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE": "module.update_block.mask.2.bias",
}

def main() -> None:
    # Make checkpoint reader
    reader = py_checkpoint_reader.NewCheckpointReader(args.tf_checkpoint)
    # Get dict of weights names and shapes
    var_to_shape = reader.get_variable_to_shape_map()
    # Init PyTorch state dict
    state_dict = {}
    # Convert weights
    for key, value in sorted(var_to_shape.items()):
        if key in TFtoPT.keys():
            if "kernel" in key:
                state_dict[TFtoPT[key]] = torch.from_numpy(reader.get_tensor(key)).permute(3, 2, 0, 1)
            else:
                state_dict[TFtoPT[key]] = torch.from_numpy(reader.get_tensor(key))
    # Save checkpoint
    torch.save(state_dict, args.pt_checkpoint_path)


if __name__ == "__main__":
    main()
